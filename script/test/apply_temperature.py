import torch
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM, Qwen2MoeDecoderLayer
from typing import cast

# --- 1. 配置 ---

# 输入：你原始的、未经修改的模型路径
ORIGINAL_MODEL_PATH = "/root/fsas/zhanghongyu/LAMoE/models/Qwen/expert_svd_router_avg_k45"

# 输出：保存校准后模型的新路径
CALIBRATED_MODEL_PATH = "/root/fsas/zhanghongyu/LAMoE/models/Qwen/expert_svd_router_avg_k45_temp_calibrated"

# 参数：从你的 search_temp_arc.py 脚本中找到的最佳温度
# 假设你找到的最佳温度是 1.45，请在这里修改
BEST_TEMPERATURE = 1.35 

# --- 2. 主逻辑 ---

def main():
    print(f"🚀 开始应用温度校准...")
    print(f"源模型: {ORIGINAL_MODEL_PATH}")
    print(f"目标温度: {BEST_TEMPERATURE}")
    print(f"输出路径: {CALIBRATED_MODEL_PATH}")

    if not (BEST_TEMPERATURE > 0):
        print("❌ 错误: 温度必须是正数。")
        return

    # --- 加载模型和分词器 ---
    print("\n加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_PATH, trust_remote_code=True, local_files_only=True)
    model = cast(Qwen2MoeForCausalLM, AutoModelForCausalLM.from_pretrained(
        ORIGINAL_MODEL_PATH,
        dtype=torch.bfloat16, # 保持和原模型一致
        device_map="cpu",     # 在CPU上加载以修改权重，避免GPU内存问题
        trust_remote_code=True,
        local_files_only=True,
    ))
    model.eval()

    # --- 应用温度缩放 ---
    print(f"\n正在将温度 T={BEST_TEMPERATURE} 应用到 MoE 层的 router...")
    
    # 动态获取项目根目录并添加到 sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(project_root)
    from src import config as AppConfig

    num_layers_changed = 0
    for i, layer in enumerate(model.model.layers):
        if i in AppConfig.TARGET_LAYERS:
            try:
                router_module = cast(Qwen2MoeDecoderLayer, layer).mlp.gate
                
                # 直接修改 gate 层的权重和偏置
                # logits' = logits / T  等价于 W' = W / T, b' = b / T
                with torch.no_grad():
                    router_module.weight.data /= BEST_TEMPERATURE
                    if router_module.bias is not None:
                        router_module.bias.data /= BEST_TEMPERATURE
                
                print(f"  - 已校准第 {i} 层 router。")
                num_layers_changed += 1
            except AttributeError:
                print(f"⚠️ 警告: 无法在第 {i} 层找到 'mlp.gate'，跳过该层。")

    if num_layers_changed == 0:
        print("❌ 错误: 没有对任何层进行修改。请检查 `config.TARGET_LAYERS` 配置。")
        return

    print(f"\n✅ 成功校准了 {num_layers_changed} 个 MoE 层。")

    # --- 保存修改后的模型和分词器 ---
    print(f"\n正在保存校准后的模型到: {CALIBRATED_MODEL_PATH}")
    os.makedirs(CALIBRATED_MODEL_PATH, exist_ok=True)
    
    model.save_pretrained(CALIBRATED_MODEL_PATH)
    tokenizer.save_pretrained(CALIBRATED_MODEL_PATH)

    print("\n🎉 校准版模型保存完毕！")
    print("\n下一步:")
    print("在你的 lm-evaluation-harness 命令中，使用下面的路径作为 model_name：")
    print(f"  --model_name {CALIBRATED_MODEL_PATH}")


if __name__ == "__main__":
    main()