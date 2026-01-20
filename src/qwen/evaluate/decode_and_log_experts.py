import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import Dict, List, cast
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM

# 导入项目配置
from ...config import (
    CURRENT_MODEL_PATH,
    MODEL_FULL_NAME,
    EVALUATE_DIR,
    TARGET_LAYERS,
    MODEL_FULL_DIR as BASE_MODEL_NAME,
    CURRENT_MODEL_PATH as BASE_MODEL_PATH,
)

# =========================================================
# 1. 超参数 (Hyperparameters)
# =========================================================
PROMPT = "Once upon a time, in a land far, far away,"
NUM_DECODE_TOKENS = 1024
BATCH_SIZE = 1 # 固定为 1

# =========================================================
# 2. 核心功能：解码并记录专家选择
# =========================================================
def decode_and_log_expert_activations(model, tokenizer, num_tokens_to_generate: int):
    """
    使用 bs=1 进行解码，并逐个 token 记录下每层激活的专家 ID。
    """
    model.eval()
    device = model.device

    # 准备输入
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    input_ids = inputs.input_ids

    # 用于存储所有 token 的专家选择情况
    # 结构: {layer_idx: [token_1_experts, token_2_experts, ...]}
    # 其中 token_i_experts 是一个包含 top-k 个专家 ID 的列表
    all_expert_selections = {layer: [] for layer in TARGET_LAYERS}

    # 获取 top_k 参数
    top_k = model.config.num_experts_per_tok

    expert_maps = {}
    for layer_idx in TARGET_LAYERS:
        try:
            expert_map = model.model.layers[layer_idx].mlp.expert_map
            expert_maps[layer_idx] = expert_map.to(device)
        except AttributeError:
            print(f"Warning: Could not find expert_map for layer {layer_idx}. Skipping.")


    print(f"🚀 Starting generation for {num_tokens_to_generate} tokens with bs={BATCH_SIZE}...")
    print(f"Model will select top-{top_k} experts for each token.")
    with torch.no_grad():
        # 使用 tqdm 创建进度条
        pbar = tqdm(range(num_tokens_to_generate), desc="Generating tokens")
        for _ in pbar:
            # 关键：设置 output_router_logits=True 来获取路由器的输出
            outputs = model(
                input_ids,
                output_router_logits=True,
                use_cache=True, # 在生成时必须使用 cache
            )

            # 1. 获取下一个 token 的 logits 并生成新的 token
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            # 2. 记录当前 token 的专家选择
            # outputs.router_logits 是一个元组，包含模型所有 MoE 层的 router_logits
            # 每个元素的形状: (batch_size, sequence_length, num_experts)
            router_logits_per_layer = outputs.router_logits

            

            for layer_idx in TARGET_LAYERS:
                # 我们只关心最新生成的那个 token 的选择情况
                # router_logits 的形状是 (1, seq_len, num_experts)，我们取最后一个 token
                # last_token_router_logits = router_logits_per_layer[layer_idx][0, -1, :]
                layer_router_logits = router_logits_per_layer[layer_idx]
                if layer_router_logits.dim() == 3:
                    last_token_router_logits = layer_router_logits[0, -1, :]
                else:
                    last_token_router_logits = layer_router_logits[-1, :]

                # 计算 top-k 专家
                _, selected_experts_original = torch.topk(last_token_router_logits, top_k)
                expert_map = expert_maps[layer_idx]
                selected_experts_final = expert_map[selected_experts_original]
                
                # 记录专家 ID
                all_expert_selections[layer_idx].append(selected_experts_final.cpu().tolist())

            # 3. 更新 input_ids 以进行下一次迭代
            input_ids = torch.cat([input_ids, next_token], dim=-1)

    print("✅ Generation and logging complete.")
    return all_expert_selections

# ORIGINAL_MODEL_PATH = "/root/fsas/zhanghongyu/LAMoE/models/Qwen/Qwen1.5-MoE-A2.7B/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"
# ORIGINAL_MODEL_PATH = "/root/fsas/zhanghongyu/LAMoE/models/Qwen/expert_svd_router_redierct_k45"
ORIGINAL_MODEL_PATH = "/root/fsas/zhanghongyu/LAMoE/models/Qwen/expert_svd_router_redierct_k30"


# =========================================================
# 3. 主函数
# =========================================================
if __name__ == "__main__":
    print(f"Loading model: {MODEL_FULL_NAME}")
    
    # 加载模型和分词器
    # tokenizer = AutoTokenizer.from_pretrained(CURRENT_MODEL_PATH)
    # model = AutoModelForCausalLM.from_pretrained(
    #     CURRENT_MODEL_PATH,
    #     torch_dtype=torch.bfloat16, # 根据需要调整
    #     device_map="auto",
    #     trust_remote_code=True,
    # )

    # model = cast(Qwen2MoeForCausalLM, AutoModelForCausalLM.from_pretrained(
    #     BASE_MODEL_NAME,
    #     cache_dir = BASE_MODEL_PATH,
    #     dtype=torch.bfloat16,
    #     device_map="auto", 
    #     trust_remote_code=True,
    #     local_files_only=True
    # ))

    # tokenizer = AutoTokenizer.from_pretrained(
    #     BASE_MODEL_NAME, 
    #     cache_dir = BASE_MODEL_PATH,
    #     trust_remote_code=True,
    # )

    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_PATH, trust_remote_code=True, local_files_only=True)
    model = cast(Qwen2MoeForCausalLM, AutoModelForCausalLM.from_pretrained(
        ORIGINAL_MODEL_PATH,
        dtype=torch.bfloat16, # 保持和原模型一致
        device_map="auto",     # 在CPU上加载以修改权重，避免GPU内存问题
        trust_remote_code=True,
        local_files_only=True,
    ))

    # 执行解码和日志记录
    expert_selections = decode_and_log_expert_activations(model, tokenizer, NUM_DECODE_TOKENS)

    # === 结果分析与保存 ===
    print("\n--- Analysis of Expert Activations ---")
    total_activated_experts_per_layer = {}
    for layer_idx, selections in expert_selections.items():
        # 将所有 token 的专家选择压平到一个集合中，以计算唯一激活的专家数量
        activated_experts = set(expert_id for token_experts in selections for expert_id in token_experts)
        total_activated_experts_per_layer[layer_idx] = sorted(list(activated_experts))
        
        print(f"Layer {layer_idx}:")
        print(f"  - Total unique experts activated: {len(activated_experts)}")
        # print(f"  - Activated expert IDs: {sorted(list(activated_experts))}")


    # === 保存为 JSON 文件 ===
    output_dir = os.path.join(EVALUATE_DIR, "generation_expert_logs")
    os.makedirs(output_dir, exist_ok=True)

    # 1. 保存详细的逐 token 记录
    detailed_log_path = os.path.join(output_dir, "expert_selections_per_token.json")
    with open(detailed_log_path, "w") as f:
        json.dump(expert_selections, f, indent=2)
    print(f"\n[Saved] Detailed per-token expert selections -> {detailed_log_path}")

    # 2. 保存唯一激活专家的总结
    summary_log_path = os.path.join(output_dir, "unique_activated_experts_summary.json")
    with open(summary_log_path, "w") as f:
        # 为了可读性，将 value 也转为 str
        summary_data = {
            "total_unique_experts_per_layer": {k: len(v) for k, v in total_activated_experts_per_layer.items()},
            "activated_expert_ids_per_layer": total_activated_experts_per_layer,
        }
        json.dump(summary_data, f, indent=2, sort_keys=True)
    print(f"[Saved] Summary of unique activated experts -> {summary_log_path}")