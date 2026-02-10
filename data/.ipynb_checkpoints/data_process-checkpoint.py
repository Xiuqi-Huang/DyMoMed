import json
from openai import OpenAI  # 确保安装了 openai >= 1.0.0
import httpx

http_client = httpx.Client(verify=False)
client = OpenAI(
    api_key="sk-sgQIJ9TOzf9ikrSYhNHlUTuIP9vVzBvljVwTU7SwS9YYX3wP", 
    # 注意：新版中 base_url 通常需要包含 /v1 路径
    base_url="https://api.zjuqx.cn/v1" 
)



def generate_prompt(index, case_json):
    """
    构建 Prompt，强制要求返回 Python 列表格式。
    """
    return f"""
    You are a medical assistant. Restructure the following case into a Python list format.
    
    Requirements:
    1. Output strictly a Python list: [id, history, examination_results, diagnosis]
    2. Language: English
    3. No Markdown, no code blocks, just the raw list string.
    4. Definitions:
       - id: {index}
       - history: Summary of Demographics, History, Past_Medical_History, Social_History.
       - examination_results: Summary of Physical_Examination_Findings, Test_Results.
       - diagnosis: Content from 'Correct_Diagnosis'.

    Input Case:
    {json.dumps(case_json)}
    """

def process_file(input_path, output_path, max_records=None):
    """
    处理文件并保存结果。
    :param max_records: 要处理的最大条数 (整数)，设为 None 则处理所有。
    """
    processed_data = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        # 读取所有行，过滤空行
        lines = [line for line in f if line.strip()]
        
    # 如果指定了数量，则截取
    if max_records is not None:
        lines = lines[:max_records]
        print(f"Processing first {max_records} records...")

    for idx, line in enumerate(lines):
        try:
            case_data = json.loads(line)
            prompt = generate_prompt(idx, case_data)
            
            # 2. 调用大模型 (OpenAI v1.x 新版写法)
            response = client.chat.completions.create(
                model="gpt-4o-mini", # 或 gpt-3.5-turbo
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                stream=False # 显式关闭流式输出，防止 Generator 报错
            )
            
            # 3. 获取内容 (新版使用点号访问属性)
            content = response.choices[0].message.content.strip()
            
            # 清理可能存在的 Markdown 标记 (以防万一)
            if content.startswith("```"): 
                content = content.replace("```json", "").replace("```python", "").replace("```", "").strip()

            # 将字符串转为列表
            structured_case = eval(content)
            processed_data.append(structured_case)
            print(f"[{idx+1}/{len(lines)}] Success")
            
        except Exception as e:
            print(f"[{idx+1}] Error: {e}")

    # 4. 保存结果
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(processed_data, f_out, indent=4, ensure_ascii=False)
    print(f"\nSaved {len(processed_data)} records to {output_path}")

# --- 运行示例 ---
if __name__ == "__main__":
    # max_records=5 表示只处理前5条，设为 None 则处理全部
    process_file("data/agentclinic.jsonl", "processed_cases.json", max_records=5)