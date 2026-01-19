import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import json
import os
import sys
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock, Qwen2MoeMLP, Qwen2MoeForCausalLM, Qwen2MoeDecoderLayer
from typing import cast, List, Dict, Tuple

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from src import config

# --- 1. 配置 ---
MODEL_NAME = "Qwen/expert_svd_router_avg_k45"
MODEL_PATH = "/root/fsas/zhanghongyu/LAMoE/models/Qwen/expert_svd_router_avg_k45"
OUTPUT_DIR = os.path.join(config.EVALUATE_DIR, "calibration_results")

# 验证的配置
VALIDATION_DATASET = "ai2_arc"
VALIDATION_SUBSET = "ARC-Challenge"
NUM_VALIDATION_SAMPLES = 1024
BATCH_SIZE = 1 # 根据您的 GPU 显存调整
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 搜索范围 (温度)
# T < 1.0 会使分布更 "尖锐" (sharper)
# T > 1.0 会使分布更 "平坦" (flatter)
TEMP_RANGE = np.arange(0.21, 2.21, 0.02).tolist() 

# --- 2. 辅助函数和类 ---
class RouterTemperatureHook:
    """
    一个 PyTorch Hook 类，用于在 router 的 forward pass 后通过温度缩放 logits。
    """
    def __init__(self):
        self.temperature = 1.0

    def set_temperature(self, temperature: float):
        # 避免除以零
        self.temperature = max(temperature, 1e-9)

    def __call__(self, module, input, output):
        # output 是 router 的原始 logits
        original_logits = output
        
        # 应用温度缩放
        adjusted_logits = original_logits / self.temperature
        
        return adjusted_logits

def prepare_arc_for_tcll(dataset_name, subset, num_samples, tokenizer):
    """
    加载并预处理 ARC 数据集，为 TCLL 计算做准备。
    """
    dataset = load_dataset(
        path=dataset_name,
        name=subset,
        split="train",
    ).select(range(num_samples))

    processed_samples = []
    for item in dataset:
        question = item['question']
        choices = item['choices']
        answer_key = item['answerKey']

        prompt = f"Question: {question}\nChoices:\n"
        for i, (label, text) in enumerate(zip(choices['label'], choices['text'])):
            prompt += f"{label}. {text}\n"

        prompt += "Answer:"
        
        correct_choice_label = answer_key
        # 加一个空格前缀 ' A' 以匹配模型生成习惯
        target_token_id = tokenizer.encode(f" {correct_choice_label}")[0]

        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        processed_samples.append({
            "input_ids": input_ids,
            "target_token_id": target_token_id
        })
        
    return processed_samples

# --- 3. 主逻辑 ---

def main():
    print("🚀 开始 Router Logits 温度搜索流程...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 步骤 1: 加载模型和分词器 ---
    print(f"加载模型: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    model = cast(Qwen2MoeForCausalLM, AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    ))
    model.eval()
    
    # --- 步骤 2: 为每个目标层注册 Hook ---
    hooks = []
    handles = []
    print("\n🔧 为每个目标 MoE 层注册温度缩放 Hook...")
    layers_to_process = config.TARGET_LAYERS
    if not layers_to_process:
        print("❌ 错误: config.TARGET_LAYERS 为空，请指定要分析的层。")
        return

    for i, layer in enumerate(model.model.layers):
        if i in layers_to_process:
            try:
                router_module = cast(Qwen2MoeDecoderLayer, layer).mlp.gate
                hook = RouterTemperatureHook()
                handle = router_module.register_forward_hook(hook)
                
                hooks.append(hook)
                handles.append(handle)
                print(f"  - 已在第 {i} 层注册 Hook。")
            except AttributeError:
                print(f"⚠️ 警告: 无法在第 {i} 层找到 'mlp.gate'，跳过该层。")
    
    if not handles:
        print("❌ 错误: 未能成功注册任何 Hook。")
        return

    # --- 步骤 3: 准备验证集 ---
    print("\n📚 准备 ARC 验证集用于 TCLL 评估...")
    validation_data = prepare_arc_for_tcll(
        VALIDATION_DATASET, VALIDATION_SUBSET, NUM_VALIDATION_SAMPLES, tokenizer
    )
    
    # --- 步骤 4: Grid Search 温度并评估 TCLL ---
    print(f"\n🔍 开始在 ARC 验证集上搜索最佳温度，范围: {TEMP_RANGE}")
    results = []

    for temp in TEMP_RANGE:
        for hook in hooks:
            hook.set_temperature(temp)
        
        total_tcll = 0.0
        
        with torch.no_grad():
            for sample in tqdm(validation_data, desc=f"评估 Temp={temp:.2f}"):
                input_ids = sample["input_ids"].to(DEVICE)
                target_token_id = sample["target_token_id"]

                outputs = model(input_ids)
                last_token_logits = outputs.logits[0, -1, :]

                log_probs = F.log_softmax(last_token_logits, dim=-1)
                tcll = log_probs[target_token_id].item()
                total_tcll += tcll

        avg_tcll = total_tcll / len(validation_data)
        
        print(f"Temp: {temp:.2f} -> 平均TCLL: {avg_tcll:.4f}")
        results.append({"temperature": temp, "avg_tcll": avg_tcll})

    # --- 步骤 5: 选定最佳温度并保存结果 ---
    for handle in handles:
        handle.remove()
    print("\n✅ 所有 Hook 已被移除。")

    # TCLL 的目标是最大化，所以我们找 avg_tcll 最大的结果
    best_result = max(results, key=lambda x: x["avg_tcll"])
    print("\n🎉 温度搜索完成！")
    print(f"最佳温度 (Temperature): {best_result['temperature']:.2f}")
    print(f"最高平均 TCLL: {best_result['avg_tcll']:.4f}")

    output_file = os.path.join(OUTPUT_DIR, "calibration_results_temp.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"详细结果已保存至: {output_file}")
    
    print("\n下一步建议:")
    print(f"使用选定的 temperature = {best_result['temperature']:.2f} 参数，在你的评估脚本中进行一次完整的评估。")


if __name__ == "__main__":
    main()