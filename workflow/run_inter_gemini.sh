KEY="sk-6k0jQLTznLCGeikit2ohi1sEmaJgnPtiDhS9XDCAKsAsVJ4X"

PYTHON_SCRIPT="D:\主线\研究生\科研\医疗ai\MedAgent\workflow\interaction.py"
DOCTOR_MODEL="google/gemini-3-flash-preview"
SHORT_NAME="gemini"
# DATASET="mtmed"
TOTAL_INFERENCES=10
WORKERS=10

echo "任务开始于: $(date '+%Y-%m-%d %H:%M:%S')"
echo "CONFIG -> dataset=$DATASET, doctor_model=$DOCTOR_MODEL, total_inferences=$TOTAL_INFERENCES, workers=$WORKERS"

python $PYTHON_SCRIPT \
    --openai_api_key "$KEY" \
    --doctor_llm "$DOCTOR_MODEL" \
    --short_doctor_llm "$SHORT_NAME" \
    --doctor_base_url "https://api.zjuqx.cn/v1" \
    --doctor_api_key "$KEY" \
    --agent_dataset "mtmed" \
    --total_inferences $TOTAL_INFERENCES \
    --workers $WORKERS

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "完成"
else
    echo "出现错误"
fi

echo "任务结束于: $(date '+%Y-%m-%d %H:%M:%S')"
echo "CONFIG -> dataset=$DATASET, doctor_model=$DOCTOR_MODEL, total_inferences=$TOTAL_INFERENCES, workers=$WORKERS"

echo "任务开始于: $(date '+%Y-%m-%d %H:%M:%S')"
echo "CONFIG -> dataset=$DATASET, doctor_model=$DOCTOR_MODEL, total_inferences=$TOTAL_INFERENCES, workers=$WORKERS"

python $PYTHON_SCRIPT \
    --openai_api_key "$KEY" \
    --doctor_llm "$DOCTOR_MODEL" \
    --short_doctor_llm "$SHORT_NAME" \
    --doctor_base_url "https://api.zjuqx.cn/v1" \
    --doctor_api_key "$KEY" \
    --agent_dataset "pmc" \
    --total_inferences $TOTAL_INFERENCES \
    --workers $WORKERS

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "完成"
else
    echo "出现错误"
fi

echo "任务结束于: $(date '+%Y-%m-%d %H:%M:%S')"
echo "CONFIG -> dataset=$DATASET, doctor_model=$DOCTOR_MODEL, total_inferences=$TOTAL_INFERENCES, workers=$WORKERS"

echo "任务开始于: $(date '+%Y-%m-%d %H:%M:%S')"
echo "CONFIG -> dataset=$DATASET, doctor_model=$DOCTOR_MODEL, total_inferences=$TOTAL_INFERENCES, workers=$WORKERS"

python $PYTHON_SCRIPT \
    --openai_api_key "$KEY" \
    --doctor_llm "$DOCTOR_MODEL" \
    --short_doctor_llm "$SHORT_NAME" \
    --doctor_base_url "https://api.zjuqx.cn/v1" \
    --doctor_api_key "$KEY" \
    --agent_dataset "clinic" \
    --total_inferences $TOTAL_INFERENCES \
    --workers $WORKERS

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "完成"
else
    echo "出现错误"
fi

echo "任务结束于: $(date '+%Y-%m-%d %H:%M:%S')"
echo "CONFIG -> dataset=$DATASET, doctor_model=$DOCTOR_MODEL, total_inferences=$TOTAL_INFERENCES, workers=$WORKERS"