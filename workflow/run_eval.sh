# 设置输入和输出文件夹路径
INPUT_DIR="../result/inter/clinic"
OUTPUT_DIR="../result/eval/clinic"
PYTHON_SCRIPT="../workflow/evaluation.py"
DATASET="clinic"
MODEL="google/gemini-3-flash-preview"
KEY="to be filled"
export OPENAI_BASE_URL="to be filled"

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

echo "任务开始于: $(date '+%Y-%m-%d %H:%M:%S')"

# 循环处理文件夹下所有的 .json 文件
for FILE in "$INPUT_DIR"/*.json
do
    filename=$(basename "$FILE")

    # 构造预期的评估文件名，并检查是否存在
    eval_filename="eval_$filename"
    if [[ -f "$OUTPUT_DIR/$eval_filename" ]]; then
        echo "------------------------------------------"
        echo "跳过已评估文件: $eval_filename"
        continue
    fi

    echo "------------------------------------------"
    echo "正在评估文件: $FILE"
    
    python $PYTHON_SCRIPT \
        --result_file "$FILE" \
        --dataset "$DATASET" \
        --evaluate_llm "$MODEL" \
        --api_key "$KEY" \
        --workers 10
done

echo "所有文件评估完成"
echo "任务结束于: $(date '+%Y-%m-%d %H:%M:%S')"