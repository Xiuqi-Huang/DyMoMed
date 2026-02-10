import argparse
import os, re
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import ScenarioLoader
from patient import PatientAgent, PatientMonitorAgent
from base_doctor import DoctorAgent

# 全局锁，防止多线程写入乱码
file_lock = threading.Lock()

def run_single_case(scenario_id, scenario_loader, args, output_filename):
    # 初始化部分可能报错（如数据加载失败），这部分确实无法保存，保持在外层try中
    try:
        scenario = scenario_loader.get_scenario(id=scenario_id)
        patient_agent = PatientAgent(scenario=scenario, backend_str=args.patient_llm,
                                     base_url=args.patient_base_url, api_key=getattr(args, 'patient_api_key', None)) 
        doctor_agent = DoctorAgent(scenario=scenario, backend_str=args.doctor_llm, max_infs=args.total_inferences,
                                   base_url=args.doctor_base_url, api_key=getattr(args, 'doctor_api_key', None))
        patient_monitor_agent = PatientMonitorAgent(scenario=scenario, backend_str=args.patient_llm,
                                                    base_url=args.patient_base_url, api_key=getattr(args, 'patient_api_key', None)) 

        patient_dialogue = ""
        doctor_dialogue = ""
        dialogue_history = []
        final_diagnosis = "N/A"
        final_treatment = "N/A"
        turns = 0
        
        # 新增一个标记，记录是否中途失败
        error_info = None 

        for _inf_id in range(args.total_inferences):
            turns = _inf_id + 1
            
            # --- 将 API 调用环节包裹在内部 try 中 ---
            try:
                # 1. 病人环节
                patient_dialogue = patient_agent.inference_patient(doctor_dialogue)
                # print(f"patient: {patient_dialogue}")#debug
                patient_monitor_dialogue = patient_monitor_agent.inference_patient(patient_dialogue)
                # print(f"patient_monitor: {patient_monitor_dialogue}") #debug
                
                if patient_monitor_dialogue.strip().upper() != "OK": 
                    retry_msg = f"""
                        [Validation Error]: Non-compliant response.
                        Previous Response: "{patient_dialogue}"
                        Reason: {patient_monitor_dialogue}
                        """
                    patient_dialogue = patient_agent.inference_second_patient(doctor_dialogue, retry_msg) 
                    # print(f"patient: {patient_dialogue}") #debug

                dialogue_history.append({"role": "patient", "content": patient_dialogue, "turn": _inf_id}) 

                # 2. 医生环节
                if _inf_id == args.total_inferences - 1:
                    patient_dialogue += "\n[SYSTEM COMMAND]: This is the LAST turn. You MUST set 'ACTION: diagnose' now and provide the final 'Diagnosis' and 'Treatment'."
                
                raw_doctor_output = doctor_agent.inference_doctor(patient_dialogue)
                # print(f"doctor_raw: {raw_doctor_output}")#debug
                
                # 解析逻辑
                parsed_output = {} 
                patterns = {
                    "thought": r"(?:Thought|Thinking|Think)[:\n]\s*(.*?)(?=\s*(?:ACTION|Response|Diagnosis|Treatment|##\s*Final\s*Response):|$)",
                    "action": r"ACTION:\s*(.*?)(?=\s*(?:Thought|Thinking|Think|Response|Diagnosis|Treatment|##):|$)",
                    "response": r"(?:Response:|##\s*Final\s*Response)\s*(.*?)(?=\s*(?:Thought|Thinking|Think|ACTION|Diagnosis|Treatment):|$)",
                    "diagnosis": r"Diagnosis:\s*(.*?)(?=\s*(?:Thought|Thinking|Think|ACTION|Response|Treatment|##):|$)",
                    "treatment": r"Treatment:\s*(.*?)(?=\s*(?:Thought|Thinking|Think|ACTION|Response|Diagnosis|##):|$)"}

                for key, pattern in patterns.items():
                    match = re.search(pattern, raw_doctor_output, re.DOTALL | re.IGNORECASE)
                    if match:
                        parsed_output[key] = match.group(1).strip()
            
                doctor_thought = parsed_output.get("thought", "")
                doctor_action = parsed_output.get("action", "continue")
                doctor_response = parsed_output.get("response", raw_doctor_output).strip()
                curr_diag = parsed_output.get("diagnosis", "").strip()
                curr_treat = parsed_output.get("treatment", "").strip()

                # 将结构化字典存入历史记录
                dialogue_history.append({
                    "role": "doctor", "turn": _inf_id, "thought": doctor_thought, "action": doctor_action,
                    "content": doctor_response, "diagnosis": curr_diag, "treatment": curr_treat}) 

                # 更新传递给下一轮病人的对话以及最终结果
                doctor_dialogue = doctor_response
                # print(f"doctor: {doctor_dialogue}")#debug
                if curr_diag: final_diagnosis = curr_diag
                if curr_treat: final_treatment = curr_treat

                if doctor_action.lower() == "diagnose":
                    print(f"\n[System]: Finished. \nDiagnosis: {final_diagnosis}")
                    break
            
            except Exception as loop_e:
                # 捕获 API 错误，记录日志，跳出循环，但不抛出到函数外
                error_info = str(loop_e)
                print(f"[Warning] ID: {scenario.patient_id} Interrupted at turn {turns}. Error: {loop_e}")
                # 标记诊断为中断
                if final_diagnosis == "N/A":
                    final_diagnosis = "N/A (API_FAILED)"
                break 
        
        record = {
            "id": scenario.patient_id,
            "final_diagnosis": final_diagnosis,
            "final_treatment": final_treatment,
            "total_turns": turns,
            "doctor_tokens": doctor_agent.token_cost,
            "patient_tokens": patient_agent.token_cost, 
            "dialogue_history": dialogue_history,
            "status": error_info if error_info else "completed"
        }

        # 实时写入 JSONL
        with file_lock:
            with open(output_filename, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        # 即使部分失败，也返回 True，避免主程序反复重试这条已经“尽力保存”的数据
        print(f"[Saved] ID: {scenario.patient_id} | Turns: {turns} | Status: {record['status']}")
        return True

    except Exception as e:
        # 这里只捕获初始化失败（如 scenario_loader 报错），这种情况下没法保存
        print(f"[Critical Error] ID: {scenario_id} Init failed. Error: {e}")
        return False

def main(args):
    if args.openai_api_key: os.environ["OPENAI_API_KEY"] = args.openai_api_key

    
    # 中间文件使用 jsonl 格式（方便追加和断点续传）
    result_dir = f"../result/inter/{args.agent_dataset}"
    temp_jsonl_file = os.path.join(result_dir, f"{args.agent_dataset}_{args.short_doctor_llm}.jsonl")
    final_output_file = os.path.join(result_dir, f"{args.agent_dataset}_{args.short_doctor_llm}.json")

    if os.path.exists(final_output_file):
        print(f"检测到最终结果文件已存在: {final_output_file}")
        print("该数据集与模型的测试任务已完成，程序自动退出。")
        return #如果数据已经处理完了，就直接返回

    scenario_loader = ScenarioLoader(dataset_path=args.agent_dataset)
    total_count = args.num_scenarios if args.num_scenarios is not None else scenario_loader.num_scenarios
    all_indices = list(range(total_count))

    # --- 断点续传检查 ---
    finished_ids = set()
    if os.path.exists(temp_jsonl_file):
        print(f"检测到进度文件 {temp_jsonl_file}，正在读取断点...")
        with open(temp_jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    if line.strip(): finished_ids.add(json.loads(line)['id'])
                except: continue
    
    todos = [i for i in all_indices if scenario_loader.get_scenario(i).patient_id not in finished_ids]
    print(f"已完成: {len(finished_ids)}, 剩余: {len(todos)}")

    #并发执行
    print(f"开始执行，并发数: {args.workers}")
    completed_in_session = 0  # 本次运行完成的计数器
    total_todos = len(todos)   # 本次需要运行的总数

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_single_case, idx, scenario_loader, args, temp_jsonl_file) for idx in todos]
        for future in as_completed(futures):
            future.result()
            completed_in_session += 1
            if completed_in_session % 5 == 0:
                    total_done = len(finished_ids) + completed_in_session
                    print(f">>> [进度提示] 本次已处理: {completed_in_session}/{total_todos} | 总进度: {total_done}/{total_count}")

    # 最后转存为 JSON 列表格式
    print("所有任务完成，正在转换格式 (JSONL -> JSON)...")
    final_data = []
    if os.path.exists(temp_jsonl_file):
        with open(temp_jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    final_data.append(json.loads(line))
    
    # 按 ID 排序
    final_data.sort(key=lambda x: x.get("id", 0))

    final_output_file = temp_jsonl_file.replace(".jsonl", ".json")
    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
        
    print(f"最终结果已保存: {final_output_file}")

    if os.path.exists(temp_jsonl_file):
        os.remove(temp_jsonl_file)
        print(f"临时进度文件已清理: {temp_jsonl_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai_api_key', type=str, required=False) #api的key
    parser.add_argument('--doctor_llm', type=str, default='google/gemini-3-flash-preview') #医生使用的模型
    parser.add_argument('--short_doctor_llm', type=str, default='jidegaiming') #简写的医生模型名，用于文件命名和区分我们的模型，要记得修改
    parser.add_argument('--doctor_base_url', type=str, default='to be filled')
    parser.add_argument('--doctor_api_key', type=str, default=None)
    parser.add_argument('--patient_llm', type=str, default='google/gemini-3-flash-preview') #病人和病人监督器
    parser.add_argument('--patient_base_url', type=str, default='to be filled')
    parser.add_argument('--patient_api_key', type=str, default=None)
    parser.add_argument('--agent_dataset', type=str, default='clinic') #数据集
    parser.add_argument('--num_scenarios', type=int, default=None) #要跑多少个case
    parser.add_argument('--total_inferences', type=int, default=10) #每个case最多多少轮对话
    parser.add_argument('--workers', type=int, default=10, help='Concurrency count') #并发数
    args = parser.parse_args()
    main(args)