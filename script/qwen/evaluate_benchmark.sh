#!/bin/bash
# script/qwen/evaluate_benchmark.sh

# 设置环境变量以使用本地缓存和镜像
export HF_ENDPOINT=https://hf-mirror.com

# 获取脚本启动时的绝对路径
SCRIPT_DIR=$(pwd)

# export HF_TOKEN="*"  # 请替换为你的Hugging Face访问令牌
source $SCRIPT_DIR/secrets/hf_token.sh
huggingface-cli login --token $HF_TOKEN

LOG_DIR="$SCRIPT_DIR/script/qwen/logs"
mkdir -p $LOG_DIR

# include sitecustomize if it exists
export PYTHONPATH="/root/LAMoE:${PYTHONPATH}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RAW_LOG="$LOG_DIR/evaluate_benchmark_${TIMESTAMP}.log"
touch "$RAW_LOG"  # 创建空日志文件

CLUSTER_N=45  # 与压缩时的聚类数保持一致

# 配置
ORIGINAL_MODEL_NAME="Qwen/Qwen1.5-MoE-A2.7B-Chat"
CACHE_DIR="/root/fsas/models/Qwen/Qwen1.5-MoE-A2.7B-Chat"
# COMPRESSED_MODEL="/root/fsas/zhanghongyu/SMoE/qwen/merged_models/qwen1.5_moe_merged_svd_cluster_${CLUSTER_N}"
COMPRESSED_MODEL="/root/fsas/zhanghongyu/LAMoE/models/Qwen/expert_svd_router_avg_k45_temp_calibrated"
OUTPUT_DIR="/root/fsas/zhanghongyu/LAMoE/qwen/eval_results"


# 根据表格定义的任务
TASKS="mmlu,winogrande,arc_easy,arc_challenge"

# 评估参数
BATCH_SIZE=4
NUM_FEWSHOT=0  # 0-shot evaluation
LIMIT=1000  # 限制样本数量以加快测试，设为null则使用全部数据

echo "🚀 开始Benchmark评估..." | tee "$RAW_LOG"
echo "📝 原始日志: $RAW_LOG" | tee -a "$RAW_LOG"
echo "⏰ 开始时间: $(date)" | tee -a "$RAW_LOG"
echo "📊 任务: $TASKS" | tee -a "$RAW_LOG"
echo "🔢 批次大小: $BATCH_SIZE" | tee -a "$RAW_LOG"
echo "🎯 Few-shot: $NUM_FEWSHOT" | tee -a "$RAW_LOG"

mkdir -p "$OUTPUT_DIR"
cd lm-evaluation-harness

# 评估压缩模型
{
    echo "=================================="
    echo "📊 评估原始模型（基线对比）..."
    echo "模型路径: $ORIGINAL_MODEL_PATH"
    echo "=================================="
    
    # lm_eval --model hf \
    #     --model_args pretrained=$ORIGINAL_MODEL_NAME,trust_remote_code=True,cache_dir=$CACHE_DIR \
    #     --tasks $TASKS \
    #     --num_fewshot $NUM_FEWSHOT \
    #     --batch_size $BATCH_SIZE \
    #     --device cuda \
    #     --limit $LIMIT \
    #     --output_path $OUTPUT_DIR/original_benchmark_results.json \
    #     --log_samples
        
    echo ""
    echo "=================================="
    echo "📊 评估压缩后的模型..."
    echo "模型路径: $COMPRESSED_MODEL"
    echo "=================================="
    
    lm_eval --model hf \
        --model_args pretrained=$COMPRESSED_MODEL,trust_remote_code=True \
        --tasks $TASKS \
        --num_fewshot $NUM_FEWSHOT \
        --batch_size $BATCH_SIZE \
        --device cuda \
        --limit $LIMIT \
        --output_path $OUTPUT_DIR/compressed_benchmark_results.json \
        --log_samples


} 2>&1 | tee -a "$RAW_LOG"

cd ..

echo ""
echo "✅ Benchmark评估完成！"
echo "📁 结果保存在: $OUTPUT_DIR/"
echo "📝 完整日志: $RAW_LOG"