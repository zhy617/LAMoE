import torch
from torch import nn
from datetime import datetime

import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, cast, Any
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock, Qwen2MoeMLP, Qwen2MoeForCausalLM, Qwen2MoeDecoderLayer
from transformers import PreTrainedTokenizerBase, Qwen2TokenizerFast, AutoModelForCausalLM, AutoTokenizer
import copy
from torch.linalg import svd

from dataclasses import dataclass, field

def load_clustering_results(cluster_dir: str, layer_idx: int) -> Tuple[torch.Tensor, Dict]:
    """加载聚类结果"""
    labels_path = os.path.join(cluster_dir, f"cluster_labels_layer_{layer_idx}.pt")
    info_path = os.path.join(cluster_dir, f"cluster_info_layer_{layer_idx}.json")
    
    if not os.path.exists(labels_path) or not os.path.exists(info_path):
        raise FileNotFoundError(f"Clustering results not found for layer {layer_idx}")

    cluster_labels: torch.Tensor = torch.load(labels_path, map_location='cpu')
    with open(info_path, 'r') as f:
        cluster_info: Dict = json.load(f)

    return cluster_labels, cluster_info

def load_activation_frequency(result_dir: str, layer_idx: int) -> torch.Tensor:
    """加载专家激活频率"""
    freq_path = os.path.join(result_dir, f"activation_frequency_layer_{layer_idx}.pt")
    
    if not os.path.exists(freq_path):
        raise FileNotFoundError(f"Activation frequency not found for layer {layer_idx}")
    
    freq_data = torch.load(freq_path, map_location='cpu')
    return freq_data['activation_counts']

def svd_subspace_alignment(expert_weights: List[torch.Tensor], relative_frequencies: List[float]) -> torch.Tensor:
    """
    使用SVD进行子空间对齐 - 按照论文公式(5-7)
    
    Args:
        expert_weights: 专家权重矩阵列表，每个形状为 [hidden_size, intermediate_size]
        relative_frequencies: 对应的激活频率列表（归一化后）
        
    Returns:
        aligned_weight: 对齐后的合并权重矩阵
    """
    if len(expert_weights) == 1:
        return expert_weights[0]
    
    if len(expert_weights) != len(relative_frequencies):
        raise ValueError("Number of weights and frequencies must match")

    # 记录原始数据类型
    original_dtype = expert_weights[0].dtype
    original_device = expert_weights[0].device

    # Step 1: 转换为float32进行SVD计算，确保在同一设备上
    expert_weights_float32 = [w.float().to(original_device) for w in expert_weights]
    
    # Step 2: 将所有专家权重垂直连接 - 按照论文公式(5)
    # W^(1), W^(2), ..., W^(n) -> [W^(1); W^(2); ...; W^(n)]
    concatenated_weights = torch.cat(expert_weights_float32, dim=1)  # [hidden_size, n_experts * intermediate_size]
    
    # Step 3: 进行SVD分解: W = U Σ V^T
    svd_result = svd(concatenated_weights, full_matrices=False)
    U: torch.Tensor = svd_result[0]  # [hidden_size, min(hidden_size, n_experts * intermediate_size)]
    S: torch.Tensor = svd_result[1]  # [min(hidden_size, n_experts * intermediate_size)]
    Vt: torch.Tensor = svd_result[2] # [min(hidden_size, n_experts * intermediate_size), n_experts * intermediate_size]
    
    # 保留与单个专家相同的维度
    intermediate_size = expert_weights[0].shape[1]  # intermediate_size
    
    
    # Step 4: 将V^T矩阵分割回原来的专家块
    # V^T的形状是 [min(hidden_size, n_experts * intermediate_size), n_experts * intermediate_size]
    # 我们需要将列维度分割成 n_experts 个块，每个块大小为 intermediate_size
    V_blocks: List[torch.Tensor] = []
    for i in range(len(expert_weights)):
        start_col = i * intermediate_size
        end_col = (i + 1) * intermediate_size
        V_i = Vt[:, start_col:end_col]  # [min(hidden_size, n_experts * intermediate_size), intermediate_size]
        V_blocks.append(V_i)
    
    # Step 4: 按照激活频率对V块进行加权合并 - 论文公式(7)
    # V_merged = Σ f(V_i) * V_i / Σ f(V_i)
    # 加权合并V块
    V_merged = torch.zeros_like(V_blocks[0])  # [svd_rank, intermediate_size]
    for V_block, weight in zip(V_blocks, relative_frequencies):
        V_merged += weight * V_block
    
    # Step 5: 重构最终的权重矩阵
    # W_merged = U_reduced @ diag(S_reduced) @ V_merged^T
    aligned_weight = U @ torch.diag(S) @ V_merged  # [hidden_size, intermediate_size]
    
    # 添加类型转换
    aligned_weight = aligned_weight.to(dtype=original_dtype, device=original_device)

    return aligned_weight

def get_cluster_relative_frequencies(
    expert_frequencies: torch.Tensor,
    cluster_labels: torch.Tensor,
    cluster_id: int
) -> List[float]:
    """
    input: 
        expert_frequencies: 专家激活频率张量
        cluster_labels: 专家聚类标签张量
        cluster_id: 当前聚类的ID
    output:
        relative_frequencies: 聚类内专家的相对激活频率
    计算聚类内专家的相对激活频率
    """
    # 找到属于当前聚类的专家
    expert_indices = torch.where(cluster_labels == cluster_id)[0]
    
    # 获取聚类内专家的激活计数
    cluster_counts = expert_frequencies[expert_indices]
    
    # 在聚类内重新归一化
    total_cluster_counts = cluster_counts.sum().float()
    
    if total_cluster_counts > 0:
        # 计算聚类内的相对频率
        relative_frequencies = (cluster_counts.float() / total_cluster_counts).tolist()
    else:
        # 如果聚类内所有专家激活都为0，使用均匀分布
        num_experts = len(expert_indices)
        relative_frequencies = [1.0 / num_experts] * num_experts
        print(f"Warning: All experts in cluster {cluster_id} have zero activation, using uniform distribution")
    
    return relative_frequencies

def merge_experts_in_moe_layer(
    moe_layer: Qwen2MoeSparseMoeBlock,
    cluster_labels: torch.Tensor,
    expert_frequencies: torch.Tensor,
    merging_method: str = "svd"
) -> Qwen2MoeSparseMoeBlock:
    """
    合并MoE层中的专家 - 修复设备一致性问题
    """
    # 获取模型所在的设备
    model_device = next(moe_layer.parameters()).device

    # 确保所有张量都在同一设备上
    cluster_labels = cluster_labels.to(model_device)
    expert_frequencies = expert_frequencies.to(model_device)

    # 创建新的MoE层副本
    merged_moe_layer = copy.deepcopy(moe_layer)
    
    # 获取聚类信息
    unique_clusters: List[int] = torch.unique(cluster_labels).tolist()
    n_merged_experts = len(unique_clusters)
    
    print(f"Merging {len(moe_layer.experts)} experts into {n_merged_experts} experts using {merging_method} method")
    
    # 创建新的专家列表
    new_experts: List[Qwen2MoeMLP] = []
    
    for cluster_id in unique_clusters:
        # 找到属于当前聚类的专家
        expert_indices: List[int] = torch.where(cluster_labels == cluster_id)[0].tolist()
        
        if len(expert_indices) == 1:
            # 如果聚类中只有一个专家，直接复制
            new_experts.append(cast(Qwen2MoeMLP, copy.deepcopy(moe_layer.experts[expert_indices[0]])))
            print(f"  Cluster {cluster_id}: single expert {expert_indices[0]}")
        else:
            # 合并多个专家
            print(f"  Cluster {cluster_id}: merging experts {expert_indices}")
            
            # 提取专家权重
            cluster_experts = [moe_layer.experts[i] for i in expert_indices]
            
            # 计算聚类内的相对频率
            cluster_frequencies = get_cluster_relative_frequencies(
                expert_frequencies, cluster_labels, cluster_id
            )
            
            print(f"    Cluster relative frequencies: {[f'{f:.4f}' for f in cluster_frequencies]}")
            
            # 创建新的专家作为模板
            merged_expert: Qwen2MoeMLP = cast(Qwen2MoeMLP, copy.deepcopy(cluster_experts[0]))
            
            # 分别合并每个权重矩阵
            for param_name in ['gate_proj', 'up_proj', 'down_proj']:
                if hasattr(merged_expert, param_name):
                    # 收集当前参数的权重
                    param_weights: List[torch.Tensor] = []
                    for expert in cluster_experts:
                        param_weights.append(cast(nn.Linear,getattr(expert, param_name)).weight.data)
                    
                    # 选择合并方法
                    merged_weight = svd_subspace_alignment(param_weights, cluster_frequencies)
                    
                    # 更新合并后的权重
                    assert(merged_weight is not None)
                    cast(nn.Linear, getattr(merged_expert, param_name)).weight.data.copy_(merged_weight)
            
            new_experts.append(merged_expert)
    
    # 更新MoE层的专家列表
    merged_moe_layer.experts = torch.nn.ModuleList(new_experts)
    merged_moe_layer.num_experts = len(new_experts)
    
    if router_merging_method == "redierct":
        print("  Redirect merging method selected, skipping router weight update")
        num_original_experts = len(moe_layer.experts)
        expert_map = torch.zeros(num_original_experts, dtype=torch.long, device=model_device)
        for new_idx, cluster_id in enumerate(unique_clusters):
            expert_indices = torch.where(cluster_labels == cluster_id)[0].tolist()
            expert_map[expert_indices] = new_idx

        merged_moe_layer.register_buffer("expert_map", expert_map)
        merged_moe_layer.num_experts = n_merged_experts
        print(f"  ✅ Created and registered 'expert_map' with shape {expert_map.shape}")
    else:
        # 更新路由器的输出维度
        if hasattr(merged_moe_layer, 'gate'):
            old_gate_weight = merged_moe_layer.gate.weight.data # [num_experts, hidden_size]
            new_gate_weight = torch.zeros(n_merged_experts, old_gate_weight.shape[1])
            old_dtype = old_gate_weight.dtype
            old_device = old_gate_weight.device

            # 用于 logit 调整的 f_k 列表
            fk_list: List[float] = []
            total_counts_layer = expert_frequencies.sum().float()
            
            # 为每个新专家分配路由权重
            for new_idx, cluster_id in enumerate(unique_clusters):
                expert_indices = torch.where(cluster_labels == cluster_id)[0].tolist()
                
                # 使用激活频率作为权重来计算聚类的路由权重
                cluster_counts = expert_frequencies[expert_indices].float()
                total_cluster_counts = cluster_counts.sum()

                # 计算簇整体激活频率 f_k（相对于整层）
                if total_counts_layer > 0:
                    f_k = (total_cluster_counts / total_counts_layer).item()
                else:
                    f_k = 1.0 / n_merged_experts
                fk_list.append(f_k)
                
                # if total_cluster_counts > 0:
                #     # 按激活频率加权平均原始路由权重
                #     weights = cluster_counts / total_cluster_counts # [len(expert_indices)]
                #     new_gate_weight[new_idx] = (old_gate_weight[expert_indices] * weights.unsqueeze(1)).sum(dim=0)
                # else:
                #     # 如果没有激活，使用简单平均
                #     new_gate_weight[new_idx] = old_gate_weight[expert_indices].mean(dim=0)

                if merge_gate_with_svd and len(expert_indices) > 1:
                    # 使用 SVD 合并 gate 行向量：把每个 row 视为 [hidden_size, 1]
                    gate_row_list = []
                    for i in expert_indices:
                        gate_row = old_gate_weight[i].unsqueeze(1)  # [hidden_size,1]
                        gate_row_list.append(gate_row)
                    # cluster 内的相对频率
                    cluster_rel_freq = get_cluster_relative_frequencies(expert_frequencies, cluster_labels, cluster_id)
                    merged_gate_col = svd_subspace_alignment(gate_row_list, cluster_rel_freq)  # [hidden_size,1]
                    new_gate_weight[new_idx] = merged_gate_col.squeeze(1).to(dtype=old_dtype, device=old_device)
                else:
                    # 使用激活频率作为权重来计算聚类的路由权重（原实现）
                    if total_cluster_counts > 0:
                        weights = cluster_counts / total_cluster_counts # [len(expert_indices)]
                        new_gate_weight[new_idx] = (old_gate_weight[expert_indices] * weights.unsqueeze(1)).sum(dim=0)
                    else:
                        new_gate_weight[new_idx] = old_gate_weight[expert_indices].mean(dim=0)
            
            # 重新创建gate层
            # merged_moe_layer.gate = torch.nn.Linear(
            #     old_gate_weight.shape[1], 
            #     n_merged_experts, 
            #     bias=False
            # )
            # merged_moe_layer.gate.weight.data.copy_(new_gate_weight)

            # 重新创建gate层：如果需要 logit 调整，则启用 bias 并填充为 fk_list
            if apply_logit_adjustment:
                merged_moe_layer.gate = torch.nn.Linear(
                    old_gate_weight.shape[1],
                    n_merged_experts,
                    bias=True
                )
            else:
                merged_moe_layer.gate = torch.nn.Linear(
                    old_gate_weight.shape[1],
                    n_merged_experts,
                    bias=False
                )
            merged_moe_layer.gate.weight.data.copy_(new_gate_weight)
            
            if apply_logit_adjustment:
                bias_tensor = torch.tensor(fk_list, dtype=old_dtype, device=old_device)
                # 数值稳定性保护，避免 log(0)
                eps = 1e-12
                bias_tensor = torch.clamp(bias_tensor, min=eps)
                # 使用自然对数 ln(f_k)
                bias_tensor = torch.log(bias_tensor) * TAU_FOR_MERGED_MODEL
                # 将 bias 加入 gate.bias (o_k + ln f_k)
                merged_moe_layer.gate.bias.data.copy_(bias_tensor)
    
    return merged_moe_layer

@dataclass
class MergeStats:
    total_layers_processed: int = 0
    total_experts_before: int = 0
    total_experts_after: int = 0
    layer_details: Dict[int, Dict] = field(default_factory=dict)

def merge_model_experts(
    model: Qwen2MoeForCausalLM,
    cluster_dir: str,
    result_dir: str,
    target_layers: List[int],
    merging_method: str = "svd"
) -> Qwen2MoeForCausalLM:
    """
    合并整个模型中指定层的专家
    
    Args:
        model: 原始模型
        cluster_dir: 聚类结果目录
        result_dir: 分析结果目录（用于加载激活频率）
        target_layers: 要合并的层列表
        merging_method: 合并方法 ("svd" 或 "frequency")
        
    Returns:
        merged_model: 合并后的模型
    """
    print(f"Starting expert merging for {len(target_layers)} layers...")
    print(f"Target layers: {target_layers}")
    print(f"Merging method: {merging_method}")
    
    # 创建模型副本
    merged_model = copy.deepcopy(model)
    
    # 统计信息
    merge_stats = MergeStats()
    
    for layer_idx in target_layers:
        print(f"\n{'='*50}")
        print(f"Processing layer {layer_idx}...")
        
        # 检查是否为MoE层
        layer = cast(Qwen2MoeDecoderLayer, merged_model.model.layers[layer_idx])
        if not isinstance(layer.mlp, Qwen2MoeSparseMoeBlock):
            print(f"  ⚠️  Layer {layer_idx} is not a MoE layer, skipping...")
            continue
        
        try:
            # 记录合并前的专家数量
            experts_before = len(layer.mlp.experts)
            merge_stats.total_experts_before += experts_before
            
            # 加载聚类结果和激活频率
            print(f"  📂 Loading clustering results for layer {layer_idx}...")
            cluster_labels, cluster_info = load_clustering_results(cluster_dir, layer_idx)
            
            print(f"  📂 Loading activation frequencies for layer {layer_idx}...")
            expert_frequencies = load_activation_frequency(result_dir, layer_idx)
            
            print(f"  🔍 Layer {layer_idx} info:")
            print(f"     - Experts before: {experts_before}")
            print(f"     - Target clusters: {cluster_info['n_clusters']}")
            print(f"     - Cluster sizes: {cluster_info['cluster_sizes']}")
            
            # 合并专家
            print(f"  🔄 Merging experts using {merging_method} method...")
            merged_moe = merge_experts_in_moe_layer(
                layer.mlp, 
                cluster_labels, 
                expert_frequencies,
                merging_method
            )
            
            # 替换层
            layer.mlp = merged_moe
            
            # 记录合并后的专家数量
            experts_after = len(merged_moe.experts)
            merge_stats.total_experts_after += experts_after
            merge_stats.total_layers_processed += 1
            
            # 记录层级详细信息
            merge_stats.layer_details[layer_idx] = {
                'experts_before': experts_before,
                'experts_after': experts_after,
                'compression_ratio': experts_before / experts_after if experts_after > 0 else float('inf'),
                'cluster_sizes': cluster_info['cluster_sizes']
            }
            
            print(f"  ✅ Layer {layer_idx} merged successfully: {experts_before} -> {experts_after} experts")
            print(f"     Compression ratio: {experts_before/experts_after:.2f}x")
            
        except FileNotFoundError as e:
            print(f"  ❌ Error: Missing required files for layer {layer_idx}")
            print(f"     {str(e)}")
            continue
            
        except Exception as e:
            print(f"  ❌ Error processing layer {layer_idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 打印总结统计
    print(f"\n{'='*60}")
    print("🎉 EXPERT MERGING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully processed layers: {merge_stats.total_layers_processed}/{len(target_layers)}")
    print(f"📊 Total experts before merging: {merge_stats.total_experts_before}")
    print(f"📊 Total experts after merging: {merge_stats.total_experts_after}")

    if merge_stats.total_experts_before > 0:
        overall_compression = merge_stats.total_experts_before / merge_stats.total_experts_after
        print(f"🎯 Overall compression ratio: {overall_compression:.2f}x")
        print(f"💾 Model size reduction: {((merge_stats.total_experts_before - merge_stats.total_experts_after) / merge_stats.total_experts_before) * 100:.1f}%")

    print(f"\nPer-layer details:")
    for layer_idx, details in merge_stats.layer_details.items():
        print(f"  Layer {layer_idx}: {details['experts_before']} -> {details['experts_after']} experts ({details['compression_ratio']:.2f}x)")
    
    return merged_model


def save_merged_model(
    merged_model: Qwen2MoeForCausalLM,
    output_dir: str,
    model_name: str = "merged_model",
    save_config: bool = True,
    tokenizer: Optional[PreTrainedTokenizerBase] = None
) -> str:
    """
    保存合并后的模型
    
    Args:
        merged_model: 合并后的模型
        output_dir: 输出目录
        model_name: 模型名称
        save_config: 是否保存配置信息
        tokenizer: 分词器（可选，如果不提供则不保存分词器）
        
    Returns:
        model_path: 保存的模型路径
    """
    model_path = os.path.join(output_dir, model_name)
    os.makedirs(model_path, exist_ok=True)
    
    print(f"\n{'='*50}")
    print("💾 SAVING MERGED MODEL")
    print(f"{'='*50}")
    print(f"📁 Output directory: {model_path}")
    
    try:
        # 保存模型
        print("🔄 Saving model weights and configuration...")
        merged_model.save_pretrained(
            save_directory=model_path,
            safe_serialization=True,  # 使用SafeTensors格式
            max_shard_size="2GB"      # 分片大小
        )
        
        # 保存分词器（可选）
        if tokenizer is not None:
            print("🔄 Saving tokenizer...")
            try:
                tokenizer.save_pretrained(model_path)
                print("✅ Tokenizer saved successfully!")
            except Exception as e:
                print(f"⚠️  Warning: Failed to save tokenizer: {e}")
                print("   Model can still be used with original tokenizer")
        else:
            print("⏩ Skipping tokenizer saving (not provided)")
            print("   💡 Tip: Use the original model's tokenizer when loading this merged model")
        
        # 保存额外的配置信息
        if save_config:
            print("🔄 Saving merge configuration...")
            merge_info = {
                "merge_timestamp": str(datetime.now()),
                "original_model_type": "Qwen2MoeForCausalLM",
                "merged_layers": [],  # 这个可以在调用时填充
                "merging_method": "svd",
                "total_parameters": sum(p.numel() for p in merged_model.parameters()),
                "trainable_parameters": sum(p.numel() for p in merged_model.parameters() if p.requires_grad),
                "moe_layers_info": {},
                "apply_logit_adjustment": apply_logit_adjustment,
                "num_merged_experts": CLUSTER_N,
            }
            
            # 统计每层的专家数量
            moe_layer_info = {}
            for i, layer in enumerate(merged_model.model.layers):
                if isinstance(layer.mlp, Qwen2MoeSparseMoeBlock):
                    moe_layer_info[f"layer_{i}"] = {
                        "num_experts": len(layer.mlp.experts),
                        "is_moe_layer": True
                    }
                else:
                    moe_layer_info[f"layer_{i}"] = {
                        "is_moe_layer": False
                    }
            
            merge_info["moe_layers_info"] = moe_layer_info
            
            config_path = os.path.join(model_path, "merge_info.json")
            with open(config_path, 'w') as f:
                json.dump(merge_info, f, indent=2, default=str)
            
            print(f"📝 Merge configuration saved to: {config_path}")
        
        # 验证保存是否成功
        print("🔍 Validating saved model...")
        saved_files = os.listdir(model_path)
        required_files = ['config.json']
        
        missing_files = [f for f in required_files if f not in saved_files]
        if missing_files:
            print(f"⚠️  Warning: Missing files: {missing_files}")
        else:
            print("✅ All required files saved successfully!")
        
        # 计算模型大小
        total_size = sum(os.path.getsize(os.path.join(model_path, f)) 
                        for f in saved_files if os.path.isfile(os.path.join(model_path, f)))
        size_gb = total_size / (1024**3)
        
        print(f"📊 Model statistics:")
        print(f"   - Total parameters: {sum(p.numel() for p in merged_model.parameters()):,}")
        print(f"   - Model size on disk: {size_gb:.2f} GB")
        print(f"   - Number of files: {len(saved_files)}")
        
        print(f"✅ Model successfully saved to: {model_path}")
        return model_path
        
    except Exception as e:
        print(f"❌ Error saving model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def update_model_config(model_path, cluster_n) -> None:
    """更新模型配置文件中的专家数量"""
    import json
    
    config_path = os.path.join(model_path, "config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config: Dict[str, Any] = json.load(f)
        
        # 更新专家数量配置
        original_experts = config.get("num_experts", 60)
        config["num_experts"] = cluster_n
        
        # 确保 num_experts_per_tok 不超过 cluster_n
        config["num_experts_per_tok"] = min(config.get("num_experts_per_tok", 4), cluster_n)
        
        # 记录原始专家数量（用于追踪）
        config["_original_num_experts"] = original_experts
        config["_compression_info"] = {
            "method": "svd_clustering",
            "original_experts": original_experts,
            "compressed_experts": cluster_n,
            "compression_ratio": cluster_n / original_experts
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Updated config.json:")
        print(f"   num_experts: {original_experts} -> {cluster_n}")
        print(f"   num_experts_per_tok: {config['num_experts_per_tok']}")
    else:
        print(f"❌ Config file not found: {config_path}")

from ...config import (
        MODEL_FULL_NAME as BASE_MODEL_NAME,
        CURRENT_MODEL_PATH as BASE_MODEL_PATH,
        KMEANS_DIR as CLUSTER_DIR,
        FREQ_RESULT_DIR,
        MERGED_SAVE_DIR as OUTPUT_MODEL_DIR,
        TARGET_LAYERS,
        CLUSTER_N,
        EXPERT_MERGING_METHOD,
        OUTPUT_MODEL_NAME,
        MERGE_GATE_WITH_SVD as merge_gate_with_svd,
        APPLY_LOGIT_ADJUSTMENT as apply_logit_adjustment,
        TAU_FOR_MERGED_MODEL,
        ROUTER_MERGING_METHOD as router_merging_method,
    )

def main():
    """主函数：执行专家合并"""
    # 配置参数
    # CLUSTER_N = 30  # 聚类数量
    # BASE_MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
    # BASE_MODEL_PATH = "/root/fsas/models/Qwen/Qwen1.5-MoE-A2.7B-Chat"
    # CLUSTER_DIR = f"/root/fsas/zhanghongyu/SMoE/qwen/analysis_results/kmeans_clusters_{CLUSTER_N}"  # 聚类结果存放位置
    # FREQ_RESULT_DIR = "/root/fsas/zhanghongyu/SMoE/qwen/analysis_results/activation_frequency_results"   # 激活频率存放位置
    # OUTPUT_MODEL_DIR = "/root/fsas/zhanghongyu/SMoE/qwen/merged_models"
    
    # # 要合并的MoE层 
    # TARGET_LAYERS = list(range(24))
    # EXPERT_MERGING_METHOD = "svd"  # 可选: "svd" 或 "frequency"

    
    try:
        print("🚀 Starting Expert Merging Pipeline")
        print(f"{'='*60}")
        
        # 加载原始模型和tokenizer
        print("📥 Loading original model and tokenizer...")
        
        model = cast(Qwen2MoeForCausalLM, AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            cache_dir = BASE_MODEL_PATH,
            dtype=torch.bfloat16,
            device_map="auto", 
            trust_remote_code=True,
        ))

        print(f"✅ Model loaded: {type(model).__name__}")

        # 加载tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_NAME, 
            cache_dir = BASE_MODEL_PATH,
            trust_remote_code=True,
        )

        # 执行专家合并
        merged_model = merge_model_experts(
            model=model,
            cluster_dir=CLUSTER_DIR,
            result_dir=FREQ_RESULT_DIR,
            target_layers=TARGET_LAYERS,
            merging_method=EXPERT_MERGING_METHOD
        )
        
        # 保存合并后的模型（不保存分词器）
        # model_name = f"qwen1.5_moe_merged_{EXPERT_MERGING_METHOD}_cluster_{CLUSTER_N}"
        saved_path = save_merged_model(
            merged_model=merged_model,
            output_dir=OUTPUT_MODEL_DIR,
            model_name=OUTPUT_MODEL_NAME,
            save_config=True,
            tokenizer=tokenizer
        )

        # config_path = os.path.join(saved_path, "config.json")

        # 🚨 重要：更新配置文件
        print("🔄 Updating model configuration...")
        update_model_config(saved_path, CLUSTER_N)

        print(f"\n🎉 Expert merging pipeline completed successfully!")
        print(f"🎯 Merged model saved to: {saved_path}")
        
        
    except Exception as e:
        print(f"💥 Fatal error during merging: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()