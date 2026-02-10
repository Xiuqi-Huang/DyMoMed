#是将数据统一成我们想要的格式的，并且过滤掉不符合要求的记录，调用api完成，这时从2000+数据变成700/214

import os
import json
import time
import httpx
from openai import OpenAI, APIConnectionError, APITimeoutError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 1. 稳定性配置
os.environ['NO_PROXY'] = 'zjuqx.cn'
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

# 2. 初始化客户端
http_client = httpx.Client(verify=False, timeout=120.0)
client = OpenAI(
    api_key="sk-sgQIJ9TOzf9ikrSYhNHlUTuIP9vVzBvljVwTU7SwS9YYX3wP", 
    base_url="https://api.zjuqx.cn/v1",
    http_client=http_client 
)

# 3. 自动重试逻辑 (接受外部传入的 model)
@retry(
    stop=stop_after_attempt(4), 
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((APIConnectionError, APITimeoutError))
)
def call_openai_with_retry(prompt, model_name):
    return client.chat.completions.create(
        model=model_name, 
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

def generate_prompt(index, case_json):
    # original_diagnosis = case_json.get("OSCE_Examination", {}).get("Correct_Diagnosis", "Unknown")#针对agentclinic
    # case_title = case_json.get("title", "Unknown Title") #针对PMC
    case_title = case_json.get("diagnosis", "Unknown Title") #针对MTMed


    return f"""
    You are a medical assistant. Restructure the medical case into a standardized JSON object.
    
    Requirements:
    1. **Output Format**: A valid JSON object with keys: [id, Medical_Record, Clinical_Findings, Diagnosis, Department].
    2. **Language**: English.
    3. **Diagnosis Rule**:
       - Primary Source: Analyze the content: "{case_title}".
       - Extraction: Extract the specific disease name.
    4. **Department Selection**: Choose one and only one department from the following list based on the case:
       [Internal Medicine, Surgery, Obstetrics and Gynecology, Pediatrics, Psychiatry, Emergency Medicine, General Practice].
    5. **Atomic & Objective Style**:
       - Use short phrases (e.g., "Sharp chest pain" instead of "The patient feels a sharp pain").
       - Remove all personal pronouns (I, He, She, My, The patient).
    6. **Missing Data Rule**: If a specific item is not mentioned in the input, do NOT invent information. Leave the field as an empty string "".
    
    Structure Definitions:
    - **Medical_Record**: 
      - Demographics: Age, sex, occupation.
      - Chief_Complaint: Primary reason for visit + duration.
      - History_of_Present_Illness: Detailed symptoms, onset, and progression.
      - Background: Comprehensive synthesis of PMH, surgical history, family/social history, medications, allergies, and other contextual data (immunizations, travel, etc.).
    - **Clinical_Findings**: 
      - Physical_Examination: Vital signs and systematic physical findings.
      - Investigations: Laboratory tests, imaging (CT/MRI/X-ray), and biopsy.
    - **Diagnosis**: Use the words from the content: "{case_title}".
    - **Department**: Categorize into one of the 7 specified medical departments.

    -----
    Example Output:
    {{
      "id": {index},
      "Medical_Record": {{
        "Demographics": "30-year-old female; software developer",
        "Chief_Complaint": "Acute RLQ abdominal pain; 12-hour duration",
        "History_of_Present_Illness": "Sudden onset sharp pain; progressively worsening; nausea; no vomiting",
        "Background": "No significant PMH; non-smoker; no drug allergies; family history of hypertension"
      }},
      "Clinical_Findings": {{
        "Physical_Examination": "Temp 37.2°C; RLQ tenderness; Positive Rovsing's sign; No abdominal distension",
        "Investigations": "WBC 12,000/μL (elevated); Ultrasound shows enlarged appendix (7mm) with wall thickening"
      }},
      "Diagnosis": "Acute Appendicitis",
      "Department": "Surgery"
    }}
    -----

    Target Input:
    {json.dumps(case_json)}

    Target Output (with id = {index}):
    """
def process_cases(input_path, output_path, model_name, target_count=700, start_index=0):
    processed_data = []
    
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        all_lines = [line.strip() for line in f if line.strip()]

    total_available = len(all_lines)
    current_idx = start_index
    last_valid_original_idx = -1 

    print(f"--- Task Config ---")
    print(f"Model: {model_name}")
    print(f"Start Index: {start_index}")
    print(f"Target Count: {target_count}")
    print(f"-------------------")

    while len(processed_data) < target_count and current_idx < total_available:
        try:
            line = all_lines[current_idx]
            raw_data = json.loads(line)
            prompt = generate_prompt(current_idx, raw_data)

            # 核心调用：传入从 main 函数传下来的模型名
            response = call_openai_with_retry(prompt, model_name)
            content = response.choices[0].message.content
            
            if content:
                clean_content = content.replace("```json", "").replace("```", "").strip()
                case_structured = json.loads(clean_content)
                
                # --- 严格过滤逻辑 ---
                med_record = case_structured.get("Medical_Record", {})
                # 只要有一个字段为空 ""，any() 就会返回 True
                if any(v == "" for v in med_record.values()):
                    print(f"[-] Index {current_idx}: Discarded (Missing fields in Medical_Record).")
                else:
                    case_structured['id'] = len(processed_data)
                    processed_data.append(case_structured)
                    last_valid_original_idx = current_idx
                    print(f"[+] [{len(processed_data)}/{target_count}] Success! Original Index: {current_idx}")
            
            time.sleep(0.5)

        except Exception as e:
            print(f"[!] Index {current_idx} error: {e}")

        current_idx += 1

    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(processed_data, f_out, indent=4, ensure_ascii=False)
    
    # 最终报告
    print("\n" + "="*40)
    print(f"Status: COMPLETED")
    print(f"Total valid cases collected: {len(processed_data)}")
    if last_valid_original_idx != -1:
        print(f"Last valid record's original index: {last_valid_original_idx}")
    print("="*40)

# --- 运行示例 ---
if __name__ == "__main__":
    # 在这里统一配置所有参数
    CONFIG = {
        "input_file": "process_document/MTMed2082.jsonl",
        "output_file": "new700_MTMed.json",
        "model": "gpt-4.1",      # 模型名放在这里
        "target_num": 700,        # 目标条数
        "start_from": 0           # 从第x条开始 (索引从0开始)
    }

    process_cases(
        input_path=CONFIG["input_file"], 
        output_path=CONFIG["output_file"], 
        model_name=CONFIG["model"],
        target_count=CONFIG["target_num"],
        start_index=CONFIG["start_from"]
    )