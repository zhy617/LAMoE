import os

# =========================================================
# 1. 实验超参数 (Hyperparameters)
# =========================================================
CURRENT_CLUSTER_N = 60 # use for analysis
CLUSTER_N = 45
SAMPLE_SIZE = 128
MAX_LENGTH = 2048
TARGET_LAYERS = list(range(24))  # 要聚类的层

# 合并参数 (用于生成新模型名字)
EXPERT_MERGING_METHOD = "svd"
ROUTER_MERGING_METHOD = "redierct" # avg, avg_la, svd, svd_la, redirect
APPLY_LOGIT_ADJUSTMENT = False
MERGE_GATE_WITH_SVD = False
REDIRECT_ENABLED = True
TAU_FOR_MERGED_MODEL = 2.06

# =========================================================
# 2. [核心开关] 当前正在操作哪个模型？
# =========================================================
# 场景 A: 跑 Base 模型 (注释掉场景 B)
CURRENT_MODEL_PATH = "/root/fsas/zhanghongyu/LAMoE/models/Qwen/Qwen1.5-MoE-A2.7B"

# 场景 B: 跑 Merged 模型 (注释掉场景 A)
# CURRENT_MODEL_PATH = f"/root/fsas/zhanghongyu/LAMoE/models/Qwen/expert_{EXPERT_MERGING_METHOD}_router_{ROUTER_MERGING_METHOD}_k{CLUSTER_N}"

FAMILY_NAME = "Qwen"    # [关键] 家族名称 (Qwen, Mixtral)
# =========================================================
# 3. 自动路径推导 (不要动这里)
# =========================================================
# 提取模型文件夹名字 (例如 "Qwen1.5-MoE-A2.7B-Chat" 或 "expert_svd_router_avg_k30")
MODEL_NAME = os.path.basename(CURRENT_MODEL_PATH.rstrip("/"))
MODEL_FULL_NAME = f"{FAMILY_NAME}/{MODEL_NAME}" # use for huggingface loading
MODEL_FULL_DIR = CURRENT_MODEL_PATH  # use for local_files_only loading

# 项目根目录
ROOT_DIR = "/root/fsas/zhanghongyu/LAMoE"
DATASET_NAME = "wikitext"
DATASET_ROOT_DIR = os.path.join(ROOT_DIR, "dataset", DATASET_NAME)
DATASET_CACHE_DIR = os.path.join(ROOT_DIR, "dataset", DATASET_NAME)

# [核心] 当前模型专属的工作区 (Cache)
# 路径: /root/.../tensor_cache/Qwen1.5-MoE-A2.7B-Chat_wikitext_128/
# 如果切换了模型，这个路径会自动变，实现完美隔离
WORKSPACE_DIR = os.path.join(
    ROOT_DIR, 
    "tensor_cache", 
    FAMILY_NAME, # 如果是 Mixtral 这里手动改一下，或者写个简单逻辑提取
    f"{MODEL_NAME}_{DATASET_NAME}_{SAMPLE_SIZE}"
)

print(f"🚀 当前工作模型: {MODEL_NAME}")
print(f"📂 中间结果存至: {WORKSPACE_DIR}")

# =========================================================
# 4. 通用变量 (你的脚本直接引用这些)
# =========================================================

# 数据集
SAMPLE_INPUT_FILE = os.path.join(ROOT_DIR, "data", f"{DATASET_NAME}_calibration.json")

# [通用] 你的脚本里原本叫 HIDDEN_STATES_DIR
# 包含 sample_0/router_logits_layer_0.pt 等
HIDDEN_STATES_DIR = os.path.join(WORKSPACE_DIR, "hidden_states_cache") 

# [通用] 分析结果目录
FREQ_RESULT_DIR = os.path.join(WORKSPACE_DIR, "activation_frequency_results")
REDUNDANCY_DIR = os.path.join(WORKSPACE_DIR, "redundancy_results")
SIMILARITY_DIR = os.path.join(WORKSPACE_DIR, "similarity_results")

# [通用] 聚类目录 (自动对应 CLUSTER_N)
KMEANS_DIR = os.path.join(WORKSPACE_DIR, f"kmeans_clusters_{CLUSTER_N}")

# [通用] 最终图表输出目录
ANALYSIS_RESULT_DIR = os.path.join(WORKSPACE_DIR, "final_analysis")


# =========================================================
# 5. [仅合并脚本用] 新模型的保存路径
# =========================================================
# 只有在运行 merge.py 时，你需要知道“我要存到哪去”
OUTPUT_MODEL_NAME = f"expert_{EXPERT_MERGING_METHOD}_router_{ROUTER_MERGING_METHOD}_k{CLUSTER_N}"
MERGED_SAVE_DIR = os.path.join(
    ROOT_DIR, 
    "models", 
    "Qwen", 
)

# evaluate 文件夹
EVALUATE_DIR = os.path.join(ROOT_DIR, "evaluation", FAMILY_NAME, MODEL_NAME)

# 确保文件夹存在
def ensure_dirs():
    for p in [HIDDEN_STATES_DIR, FREQ_RESULT_DIR, REDUNDANCY_DIR, SIMILARITY_DIR, KMEANS_DIR, ANALYSIS_RESULT_DIR, EVALUATE_DIR]:
        os.makedirs(p, exist_ok=True)
    # 如果是合并操作，还要创建保存目录
    os.makedirs(MERGED_SAVE_DIR, exist_ok=True)