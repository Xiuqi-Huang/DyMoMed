import faulthandler
# 在import之后直接添加以下启用代码即可
faulthandler.enable()

import os
import json
import re
import time
import logging
import uuid
from typing import Dict, List, Optional, Any
from openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import httpx


from .constant import *
from .base_agent import *
from .utils import query_model

# 新增：导入RAG相关（用于复用检索/入库函数）
from .rag_b import RAG

# 基础配置
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 全局配置（可根据实际路径调整）
GOAL_JSON_PATH = "./goals.json"  # 目标集JSON保存路径
EXP_SAVE_DIR = "./experience_data"       # 经验临时保存目录
VDB_PERSIST_DIR = "./chroma_vdb"
GOAL_MAX_COUNT = 16



def query_model_ours(client, prompt, system_prompt,t=0.01, tries=2, timeout=20.0, max_prompt_len=2**14, clip_prompt=False):
    """统一的LLM调用函数（复用原有逻辑）"""
    valid_models = ["gpt4"]
    old_http_proxy = os.environ.get("http_proxy", "")
    old_https_proxy = os.environ.get("https_proxy", "")
    old_no_proxy = os.environ.get('NO_PROXY', '')
    
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""
    os.environ['NO_PROXY'] = ''
    model_str = 'gpt4'

    if model_str not in valid_models and "_HF" not in model_str:
        os.environ["http_proxy"] = old_http_proxy
        os.environ["https_proxy"] = old_https_proxy
        os.environ['NO_PROXY'] = old_no_proxy
        raise Exception(f"No model by the name {model_str}")

    try:
        for _ in range(tries):
            if clip_prompt and len(prompt) > max_prompt_len:
                prompt = prompt[:max_prompt_len]
            try:
                answer = ""
                total_tokens = 0
                if model_str == "gpt4":
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                        temperature=t)
                    our_res = response.choices[0].message.content.strip()
                    if response.usage:
                        total_tokens = response.usage.total_tokens
                    content = response.choices[0].message.content or ""
                    answer = re.sub(r"[ \t]+", " ", content)
                elif "HF_" in model_str:
                    raise Exception("HF models TODO")
                return our_res, total_tokens
            except Exception as e:
                logger.error(f"[API调用失败] 重试中 - {type(e).__name__}: {e}")
                time.sleep(timeout)
                continue
        raise Exception("Max retries: timeout")
    finally:
        os.environ["http_proxy"] = old_http_proxy
        os.environ["https_proxy"] = old_https_proxy
        os.environ['NO_PROXY'] = old_no_proxy

class OursB(BaseAgent):
    def __init__(self,bench_name, scenario, backend_str, max_infs=10, base_url=None, api_key=None, name: str = "Doctor", pubmed_reference_prob: float = 0.5):
        model_config = ModelConfig(model_type='gpt', model_name=backend_str)
        super().__init__(name=name, model_config=model_config)

        self.scenario = bench_name
        self.base_url = base_url
        self.api_key = api_key

        self.token_cost = 0


        #r 系统消耗总token数
        self.sys_tokens_cost = 0
        #r 和目标管理相关的token消耗
        self.goal_tokens_cost = 0



        # 替换原有RAG实例为RAGHandler（复用检索/入库函数）
        self.vdb_path = f"{VDB_PERSIST_DIR}_{bench_name}"


        #r 目标目录+经验库目录
        self.goals_path = f"{GOAL_JSON_PATH}_{bench_name}"
        self.exps_path = f"{EXP_SAVE_DIR}_{bench_name}"



        self.rag_handler = RAG(persist_directory=self.vdb_path)
        self.rag_handler.get_all_source_files_from_vdb()
        self.rag = self.rag_handler  # 保持原有变量名兼容
        self.MAX_INFS = max_infs
        self.backend = backend_str
        self.infs = 1
        self.history_dialogue = []

        # 目标筛选相关参数（复用原有）
        self.prog_goal_round = 2
        self.prog_threshold = 0.1
        self.k_opp_comp = 1
        self.score_threshold = 2
        self.top_k = 2
        self.compliant_threshold = 0.9

        # 经验筛选相关参数（新增）
        self.exp_k = 4  # 达标目标数量阈值
        self.exp_ratio = 1  # 达标目标比例阈值

        # ===== 新增：PubMed参考概率参数 =====
        self.pubmed_reference_prob = pubmed_reference_prob  # PubMed知识参考触发概率
        self.random_seed = 42  
        import random
        self.random = random.Random(self.random_seed)

        # 状态记录（复用原有）
        self.goal_progress_record = {}
        self.combine_results_record = {}
        self.original_goals_round = []
        self.filtered_goals_round = []
        self.blocked_goals_round = []
        self.top_k_goals_round = []
        self.top_k_goals = {} #r 按轮数“round_1”存top_k目标列表
        self.top_k_results = []
        self.edges = []
        self.secondary_blocked_goals = []
        self.discarded_goals = []
        self.initial_goal_list = []
        self.round_discarded_goals = {}

        # 病例标识（复用原有）
        self.case_id = str(uuid.uuid4())[:8]
        self.case_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # LLM客户端（复用原有）
        self.fuzhu_llm = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url="",
            http_client=httpx.Client(verify=True, timeout=httpx.Timeout(300.0, connect=60.0), follow_redirects=True)
        )
        self.monitor = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url="",
            http_client=httpx.Client(verify=True, timeout=httpx.Timeout(300.0, connect=60.0), follow_redirects=True)
        )

        # 初始化医生提示词（复用原有）
        self.init_doctor_prompt = f'''
    You are a seasoned physician in a clinic conducting a medical consultation and assessment of a patient. 

    Constraints:
    1. Response length: 1-3 sentences.
    2. Turn limit: Complete everything within {self.MAX_INFS} turns. (Current turn: {self.infs}/{self.MAX_INFS}).
    3. CRITICAL: On turn {self.MAX_INFS}, you MUST use ACTION: 'diagnose'.

    Rules:
    1. ACTION 'continue': Use if more information is needed. Your 'Response' must be a professional inquiry or follow-up.
    2. ACTION 'diagnose': Must be triggered once the turn limit (Current turn: {self.infs}/{self.MAX_INFS}) is hit. 'Diagnosis' and 'Treatment' fields are MANDATORY.
    3. Diagnosis Precision: Provide ONLY the standard medical name of the disease.

    Output format:
    Thought: [Clinical reasoning and decision rationale]
    ACTION: [continue | diagnose]
    Response: [Your professional utterances to the patient]
    Diagnosis: [The specific disease name if diagnosing; otherwise, omit this field]
    Treatment: [The treatment plan if diagnosing; otherwise, omit this field]
    '''
        
        self.is_finish_prompt = f'''
        Supplementary Core Rule:
        If you deem sufficient evidence has been obtained to confirm the patient's condition, you may use ACTION: 'diagnose' to provide the final diagnosis and treatment recommendations, even if the turn limit ({self.MAX_INFS}) has not been reached.
        '''

        # 初始化：加载/创建目标集JSON


    # ===================== 新增：目标集JSON初始化 =====================
    def _replace_old_goals(self, merged_nodes: List[Dict], new_valid_goals: List[Dict]) -> List[Dict]:
        """
        实现新目标替换老目标的逻辑
        规则：
        1. 若总目标数未达上限，直接添加新目标
        2. 若达上限，用高可行性新目标替换低优先级老目标
        3. 老目标优先级：按feasibility_score倒序，无分数的老目标优先级最低
        """
        # 1. 分离老目标和新目标
        original_names = {n.get("name") for n in self.rag_handler.goals.get("nodes", []) if isinstance(n, dict)}
        old_goals = [n for n in merged_nodes if n.get("name") in original_names]
        new_goals_candidate = [n for n in merged_nodes if n.get("name") not in original_names]

        # 2. 为老目标补充可行性分数（用于排序）
        for old_goal in old_goals:
            goal_name = old_goal.get("name")
            old_goal["feasibility_score"] = self.goal_metrics.get(goal_name, {}).get("feasibility_score", 0.0)

        # 3. 为新目标排序（按可行性分数倒序）
        new_goals_candidate.sort(key=lambda x: x.get("feasibility_score", 0.0), reverse=True)

        # 4. 计算可容纳的新目标数量
        available_slots = max(0, GOAL_MAX_COUNT - len(old_goals))

        final_goals = []
        if available_slots >= len(new_goals_candidate):
            # 有足够空间，全部添加
            final_goals = old_goals + new_goals_candidate
        else:
            # 空间不足，替换低优先级老目标
            # 老目标按可行性分数倒序，取前N个保留
            old_goals_sorted = sorted(old_goals, key=lambda x: x.get("feasibility_score", 0.0), reverse=True)
            retained_old_goals = old_goals_sorted[:GOAL_MAX_COUNT - len(new_goals_candidate)]
            final_goals = retained_old_goals + new_goals_candidate

        # 5. 最终截断到上限（防御性处理）
        final_goals = final_goals[:GOAL_MAX_COUNT]

        # 日志记录替换情况
        replaced_count = len(old_goals) - len([g for g in final_goals if g.get("name") in original_names])
        added_count = len([g for g in final_goals if g.get("name") not in original_names])
        logger.info(f"目标替换完成 | 替换老目标数: {replaced_count} | 新增新目标数: {added_count} | 最终目标总数: {len(final_goals)}")

        return final_goals


    # ===================== 目标筛选更新（保存为JSON） =====================
    def updated_goals(self) -> Dict[str, List[Any]]:
        """复用原有筛选逻辑，新增JSON保存 + 按用户要求优化筛选规则 + 过滤level3目标 + 目标数量上限 + 新目标替换老目标"""
        # 步骤1：加载原有预设目标集（存入时已做筛选，仅保留基础空值过滤做防御性编程）
        original_goals = self.rag_handler.goals
        # 仅做基础空值过滤（存入时已筛选有效节点/关系，无需重复复杂过滤）
        original_nodes = [n for n in original_goals.get("nodes", []) if n]
        original_edges = [e for e in original_goals.get("edges", []) if e]
    
        # 步骤2：计算动态目标可行性得分
        self.goal_metrics = {}  
        total_rounds = self.infs
        # 新增：平均提升进度阈值（用户要求0.1）
        feasibility_threshold = 0.2
        # 新增：最终进度阈值调整为0.8（用户要求）
        final_progress_threshold = 0.9
    
        for goal in self.goal_list:
            goal_name = goal.get("name", "").strip()
            if not goal_name or goal_name in [n.get("name") for n in original_nodes if isinstance(n, dict)]:
                continue
            
            if goal_name not in self.goal_progress_record:
                self.goal_metrics[goal_name] = {
                    "final_progress": 0.0,
                    "used_rounds": 0,
                    "feasibility_score": 0.0,  
                    "is_valid": False,
                    # 携带原目标的描述和分数（用户要求：复用原有值）
                    "description": goal.get("description", ""),
                    "score": goal.get("score", 0.0)
                }
                continue
            
            progress_records = self.goal_progress_record[goal_name]
            sorted_rounds = sorted(
                progress_records.items(),
                key=lambda x: int(x[0].replace("Round ", ""))
            )
            used_rounds = len(sorted_rounds)
    
            # 计算最终进度
            final_progress = sorted_rounds[-1][1][0] if used_rounds > 0 else 0.0
            
            if used_rounds <= 1:
                feasibility_score = 0.0  # 只有1轮无提升，记为0
            else:
                initial_progress = sorted_rounds[0][1][0]  # 第一轮进度
                total_increase = final_progress - initial_progress  # 总提升量
                feasibility_score = total_increase / (used_rounds - 1)  # 每轮平均提升
    
            #r 筛选条件
            is_valid = (final_progress >= final_progress_threshold) or (feasibility_score >= feasibility_threshold)
    
            self.goal_metrics[goal_name] = {
                "final_progress": round(final_progress, 2),
                "used_rounds": used_rounds,
                "feasibility_score": round(feasibility_score, 4),
                "is_valid": is_valid,
                # 携带原目标的描述和分数
                "description": goal.get("description", ""),
                "score": goal.get("score", 0.0)
            }
    
        # 筛选高可行性新目标（保留原目标的描述/分数）
        new_valid_goals = [
            {"name": goal_name, **metrics} for goal_name, metrics in sorted(
                self.goal_metrics.items(),
                key=lambda x: x[1]["feasibility_score"],
                reverse=True
            ) if metrics["is_valid"]
        ]
    
        # 步骤3：合并目标集
        merged_nodes = original_nodes.copy()
        original_names = [n.get("name") for n in original_nodes if isinstance(n, dict)]
        for new_goal in new_valid_goals:
            new_goal_name = new_goal["name"]
            if new_goal_name not in original_names:
                
                merged_nodes.append({
                    "name": new_goal_name,
                    "description": new_goal["description"],  # 原目标描述
                    "score": new_goal["score"],               # 原目标分数
                    "level": goal.get("level", "")            # 新增：携带level字段
                })
    
        # ========== 新增：目标数量上限控制 + 新目标替换老目标 ==========
        merged_nodes = self._replace_old_goals(merged_nodes, new_valid_goals)
    
        
        new_valid_edges = []
        valid_node_names = [n.get("name") for n in merged_nodes if isinstance(n, dict)]
        for edge in self.edges:
            if len(edge) != 3 or edge[2] not in ["prog", "opp", "comp"]:
                continue
            src, dst, rel_type = edge
            src = src.strip()
            dst = dst.strip()
            if rel_type == "comp":
                if src in valid_node_names or src == "":
                    new_valid_edges.append((src, dst, rel_type))
            else:
                if src in valid_node_names and dst in valid_node_names:
                    new_valid_edges.append((src, dst, rel_type))
    
        merged_edges = original_edges.copy()
        for new_edge in new_valid_edges:
            if new_edge not in merged_edges:
                merged_edges.append(new_edge)
    
        # 最终筛选
        final_nodes = []
        for node in merged_nodes:
            # 新增：过滤level3目标（兼容level 3、level3等格式）
            node_level = node.get("level", "").strip().lower()
            if node_level == "level 3" or node_level == "level3":
                continue
            
            if isinstance(node, dict) and node.get("name") in [n.get("name") for n in original_nodes]:
                final_nodes.append(node)
                continue
            if isinstance(node, dict) and node.get("name") in self.goal_metrics:
                metrics = self.goal_metrics[node.get("name")]
                if metrics.get("is_valid", False):
                    final_nodes.append(node)
    
        # ========== 再次截断到目标上限（双重保障） ==========
        final_nodes = final_nodes[:GOAL_MAX_COUNT]
    
        final_edges = []
        final_node_names = [n.get("name") for n in final_nodes if isinstance(n, dict)]
        for edge in merged_edges:
            src, dst, rel_type = edge
            if rel_type == "comp":
                if src in final_node_names or src == "":
                    final_edges.append(edge)
            else:
                if src in final_node_names and dst in final_node_names:
                    final_edges.append(edge)
    
        # 步骤4：更新并保存JSON
        updated_goal_set = {
            "nodes": final_nodes,
            "edges": final_edges
        }
        self.rag_handler.goals = updated_goal_set
        self._save_goal_json()
        logger.info(f"目标集已更新并保存至{self.goals_path}（已过滤level3目标，总数控制在{len(final_nodes)}/{GOAL_MAX_COUNT}）")
    
        return updated_goal_set

    def _save_goal_json(self):
        """保存目标集到JSON文件"""
        os.makedirs(os.path.dirname(self.goals_path), exist_ok=True)
        with open(self.goals_path, 'w', encoding='utf-8') as f:
            json.dump(self.rag_handler.goals, f, indent=4, ensure_ascii=False)

    # ===================== 经验筛选更新（合并LLM调用） =====================
    
    
    def filter_update_exps(self, bench_name: str, patient_id: str):
        #r 步骤1：筛选达标目标
        preset_goals = self.rag_handler.goals['nodes'] #r 最全的 
        preset_goals = self.initial_goal_list
        qualified_goals = []
        preset_goals_name = []
        for goal in preset_goals:
            goal_name = goal['name'].strip()
            preset_goals_name.append(goal_name)
            if not goal_name:
                continue
            
            progress_records = self.goal_progress_record[goal_name]
            latest_progress = list(progress_records.values())[-1][0]
            if latest_progress >= self.compliant_threshold:
                qualified_goals.append(goal_name)
    
        # 未达标则直接返回
        logger.info(f"预设目标数：{len(preset_goals)}\n包括：{preset_goals_name}")
        logger.info(f"经验达标目标数{len(qualified_goals)}/{self.exp_k}，比例{len(qualified_goals)/len(preset_goals):.2f}/{self.exp_ratio}")
        if len(qualified_goals) < self.exp_k and (len(qualified_goals) / len(preset_goals) < self.exp_ratio):
            logger.info(f"经验筛选未达标")
            return
    
        # 步骤2：处理每轮数据
        rounds_data = []
        for idx, round_dialog in enumerate(self.history_dialogue):
            round_id = idx + 1
            current_dialogue = f"Patient: {round_dialog['patient']}\nDoctor: {round_dialog['doctor']}"
            current_reasoning = self.combine_results_record[round_id]
            current_evaluation = self.round_top_k_evaluation(round_id)
            
            
            round_top_k_goal_names = self.top_k_goals[f"round_{round_id}"]
            
           
            
            # ========== 构造轮次数据，补充new_goals ==========
            rounds_data.append({
                "round_id": round_id,
                "previous_summary": "",
                "current_dialogue": current_dialogue,
                "current_reasoning": current_reasoning,
                "current_evaluation": current_evaluation,
                "new_goals": round_top_k_goal_names  # 每轮新增：本轮top_k目标名称列表
            })
    
        # 步骤3：合并LLM调用
        summary_and_eval = self._generate_summary_and_evaluation(rounds_data)
        all_rounds_summary = summary_and_eval.get("summaries", {})
        # 补充previous_summary（复用原有逻辑）
        for i in range(len(rounds_data)):
            if i == 0:
                rounds_data[i]["previous_summary"] = "Round 1, no prior context available"
            else:
                rounds_data[i]["previous_summary"] = all_rounds_summary.get(f"round_{i+1}", "")
    
        # 步骤4：保存经验文件并入库RAG
        os.makedirs(self.exps_path, exist_ok=True)
        save_path = os.path.join(self.exps_path, f"{bench_name}_{patient_id}.json")
        
        # ========== 处理顶层init_goals/new_goals ==========
        
        
        init_goals = self.initial_goal_list if isinstance(self.initial_goal_list, list) else []
        
        # 2. new_goals：转为目标名称字符串列表（核心！）
        #    self.filtered_goals_round如果是字典列表，提取name；如果是名称列表，直接用
        
        new_goals = []
        for goal in self.filtered_goals_round:
            new_goals.append(goal.get("name").strip())
    
        # 构造最终经验数据（结构适配RAG入库）
        final_exp_data = {
            "init_goals": init_goals,          # 字典列表（入库时提取名称）
            "new_goals": new_goals,            # 名称字符串列表
            "rounds": rounds_data,             # 每轮包含new_goals（名称列表）
            "patient_id": patient_id,
            "bench_name": bench_name
        }
        
        # 保存经验文件
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(final_exp_data, f, indent=4, ensure_ascii=False)
    
        # 调用RAG的ingest函数入库（复用RAG逻辑，此时结构已适配）
        self.rag_handler.ingest_single_document(save_path)
        logger.info(f"经验已保存并入库向量数据库：{save_path}")
    
    def _generate_summary_and_evaluation(self, rounds_data: List[Dict]) -> Dict:
        """仅完成任务1：为每轮生成累计previous_summary（移除评估相关逻辑）"""
        prompt_template = """
        You are a senior clinical assessment expert. Complete the following task (strict JSON format):

        TASK: Generate cumulative previous_summary for each round (except round 1)
        Rules:
        1. round_2 summary = summarize round 1 (dialogue + reasoning) → max 100 words, concise and accurate
        2. round_3 summary = round_2 summary + summarize round 2 content → max 100 words
        3. Round 1 has no prior context, so no need to generate summary for it
        4. Output ONLY JSON, no extra text/explanation

        Input Data:
        - Case ID: {case_id}
        - Round data (dialogue + reasoning): {rounds_data_str}

        Output Format (STRICT JSON):
        {{
            "summaries": {{
                "round_2": "summary text for round 2 (based on round 1)",
                "round_3": "summary text for round 3 (based on round 1+2)",
                ...
            }}
        }}
        """
        # 构造入参（仅保留对话+推理，移除评估相关）
        rounds_data_str = json.dumps(
            [{k: v for k, v in rd.items() if k in ["round_id", "current_dialogue", "current_reasoning"]} 
             for rd in rounds_data], 
            indent=2, ensure_ascii=False
        )

        prompt = prompt_template.format(
            case_id=self.case_id,
            rounds_data_str=rounds_data_str
        )

        # LLM调用（仅生成总结，简化逻辑）
        try:
            response = self.fuzhu_llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                timeout=60
            )
            #r 和目标无关
            self.sys_tokens_cost += response.usage.total_tokens


            result = json.loads(response.choices[0].message.content.strip())
            # 兜底：确保返回结构包含summaries（避免KeyError）
            return {"summaries": result.get("summaries", {})}
        except Exception as e:
            logger.error(f"生成累计总结失败：{e}")
            return {"summaries": {}}  # 异常时返回空总结，不中断主流程
        # ===================== 复用：轮次评估（适配合并LLM调用） =====================
    
    

    #r 注意只在最后才调用这个函数
    def round_top_k_evaluation(self, idx) -> str:
        round_name = f"Round {idx}"
        eval_info = []
        for goal_name, progress_records in self.goal_progress_record.items():
            if goal_name in [g.get("name") for g in self.top_k_goals[f"round_{idx}"]] and round_name in progress_records:
                progress, is_top_k, desc = progress_records[round_name][:3]
                eval_info.append(f"Goal[{goal_name}]: Progress={progress:.2f}, Description={desc}")
        return "\n".join(eval_info)



    # ===================== 复用：RAG检索函数调用 =====================
    def retrieve_for_reasoning(self, query_dict: Dict) -> Dict[str, List[Document]]:
        """复用RAG的推理时经验检索"""
        return self.rag_handler.retrieve_for_goal_reasoning(query_dict)

    def retrieve_for_response(self, previous_dialogue: str) -> List[Document]:
        """复用RAG的回复时经验检索"""
        return self.rag_handler.retrieve_for_response(previous_dialogue)

    
    def _log_info(self, content: str, level: str = "info"):
        prefix = f"[病例-{self.case_id}][轮次-{self.infs}]"
        full_content = f"{prefix} {content}"
        if level == "info":
            logger.info(full_content)
        elif level == "warning":
            logger.warning(full_content)
        elif level == "error":
            logger.error(full_content)
        elif level == "debug":
            logger.debug(full_content)

    def _log_case_start(self):
        print(f"==============病人{self.case_id}==============")
        print("="*80)
        print(f"【病例开始】ID: {self.case_id} | 开始时间: {self.case_start_time} | 最大轮次: {self.MAX_INFS}")
        print("="*80)

    def _log_case_end(self):
        case_end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print("="*80)
        print(f"【病例结束】ID: {self.case_id} | 结束时间: {case_end_time} | 总轮次: {self.infs-1} | 总Token消耗: {self.token_cost}")
        print("="*80)

    def reset_agent_state(self):
        self._log_info("【状态重置】清空所有历史数据，准备新病例", "warning")
        self.infs = 1
        self.history_dialogue.clear()
        self.goal_progress_record.clear()
        self.combine_results_record.clear()
        self.original_goals_round.clear()
        self.filtered_goals_round.clear()
        self.blocked_goals_round.clear()
        self.top_k_goals_round.clear()
        # 新增：清空按轮数存储的top_k目标字典
        self.top_k_goals.clear()  # 或 self.top_k_goals = {}
        self.top_k_results.clear()
        self.edges.clear()
        self.secondary_blocked_goals.clear()
        self.discarded_goals.clear()
        self.initial_goal_list = []
        self.round_discarded_goals = {}
        # 新增：重置累计Token消耗
        self.token_cost = 0
        self.sys_tokens_cost = 0
        self.goal_tokens_cost = 0
        # 重置病例标识
        self.case_id = str(uuid.uuid4())[:8]
        self.case_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self._log_case_start()

    def _init_missing_goal_progress(self):
        current_round = f"Round {self.infs}"
        new_goal_count = 0
        for goal in self.goal_list:
            goal_name = goal.get('name', '')
            if not goal_name:
                continue
            if 'is_progressive' not in goal:
                goal['is_progressive'] = False
            if goal_name not in self.goal_progress_record:
                self.goal_progress_record[goal_name] = {
                    current_round: (0.0, False, "New goal initialized", "normal")
                }
                new_goal_count += 1
            if current_round not in self.goal_progress_record[goal_name]:
                last_progress = 0.0
                last_status = "normal"
                old_rounds = list(self.goal_progress_record[goal_name].keys())
                if old_rounds:
                    last_round = old_rounds[-1]
                    last_progress = self.goal_progress_record[goal_name][last_round][0]
                    last_status = self.goal_progress_record[goal_name][last_round][3]
                self.goal_progress_record[goal_name][current_round] = (last_progress, False, "Inherited from last round", last_status)
        if new_goal_count > 0:
            self._log_info(f"初始化新目标进度 {new_goal_count} 个，累计进度记录: {len(self.goal_progress_record)}")

    def process_history_dialogs(self, dialogs: List[Dict], window_setting: Optional[int] = None) -> str:
        if not dialogs:
            return ""
        if window_setting is not None:
            if window_setting < 0:
                filtered_dialogs = dialogs[-abs(window_setting):] if len(dialogs) >= abs(window_setting) else dialogs
            else:
                filtered_dialogs = dialogs[:window_setting] if len(dialogs) >= window_setting else dialogs
        else:
            filtered_dialogs = dialogs
        dialog_str_list = []
        for idx, dialog in enumerate(filtered_dialogs, 1):
            patient_content = dialog.get("patient", "").strip()
            doctor_content = dialog.get("doctor", "").strip()
            dialog_str = f"Dialog {idx}\nPatient: {patient_content}\nDoctor: {doctor_content}"
            dialog_str_list.append(dialog_str)
        return "\n\n".join(dialog_str_list).strip()

    def blocked_goals(self) -> List[Dict]:
        blocked_goals = []
        self.secondary_blocked_goals.clear()
        current_topk_names = [g.get("name","") for g in self.top_k_goals_round]

        for goal_name, round_records in self.goal_progress_record.items():
            goal = next((g for g in self.goal_list if g.get('name') == goal_name), None)
            if not goal:
                continue
            is_progressive = goal.get('is_progressive', False)

            if len(round_records) < self.prog_goal_round:
                continue
            try:
                round_items = sorted(round_records.items(), key=lambda x: int(x[0].replace("Round ", "")))
            except (ValueError, KeyError):
                continue
            latest_rounds = round_items[-self.prog_goal_round:]
            if len(latest_rounds) < 2:
                continue
            
            try:
                round_1_name, (round_1_progress, round_1_is_topk, _, round_1_status) = latest_rounds[0]
                round_2_name, (round_2_progress, round_2_is_topk, _, round_2_status) = latest_rounds[1]
            except (IndexError, TypeError):
                continue
            
            if not (round_1_is_topk and round_2_is_topk) or goal_name not in current_topk_names:
                continue

            progress_increase = round_2_progress - round_1_progress
            if progress_increase >= self.prog_threshold:
                continue

            if is_progressive:
                if goal_name not in self.discarded_goals:
                    self.secondary_blocked_goals.append(goal_name)
                    self.discarded_goals.append(goal_name)
                    self._log_info(f"终阻目标加入抛弃列表: {goal_name}", "warning")
            else:
                blocked_goals.append({
                    "goal_name": goal_name,
                    round_1_name: {"progress": round_1_progress, "desc": latest_rounds[0][1][2]},
                    round_2_name: {"progress": round_2_progress, "desc": latest_rounds[1][1][2]},
                    "progress_increase": progress_increase,
                })
        self.blocked_goals_round = blocked_goals
        self._log_info(f"受阻目标识别完成 | 一级受阻: {len(blocked_goals)} | 二级终阻: {len(self.secondary_blocked_goals)}")
        return blocked_goals

    def parse_goal(self, llm_output: str) -> Dict:
        llm_output = re.sub(r'\r\n', '\n', llm_output)
        llm_output = re.sub(r'\n+', '\n', llm_output)

        name_match = re.search(r'name\s*:\s*(.+?)(?=\n|description|score|level|$)', llm_output, re.DOTALL | re.IGNORECASE)
        desc_match = re.search(r'description\s*:\s*(.+?)(?=\n|score|name|level|$)', llm_output, re.DOTALL | re.IGNORECASE)
        score_match = re.search(r'score\s*:\s*(.+?)(?=\n|name|description|level|$)', llm_output, re.DOTALL | re.IGNORECASE)
        # 新增：提取level字段（兼容level 1/2/3、Level 1等格式）
        level_match = re.search(r'level\s*:\s*(level\s*\d+)(?=\n|name|description|score|$)', llm_output, re.DOTALL | re.IGNORECASE)

        goal_name = name_match.group(1).strip() if name_match else "Unnamed Goal"
        goal_desc = desc_match.group(1).strip() if desc_match else "No description"
        score_str = score_match.group(1).strip() if score_match else "0"
        score_clean = re.sub(r'[^\d.]', '', score_str)
        try:
            goal_score = float(score_clean)
        except ValueError:
            goal_score = 0.0
        # 新增：处理level字段（标准化为小写，去除多余空格）
        goal_level = level_match.group(1).strip().lower() if level_match else ""

        return {
            "name": goal_name,
            "description": goal_desc,
            "score": goal_score,
            "level": goal_level  # 新增：将level存入目标字典
        }

    def sort_goals(self) -> List[Dict]:
        
        # 原始目标：不做任何过滤、排序、截断，直接完整保留，放在最终结果最前面
        original_goals_selected = self.initial_goal_list.copy()  # 完整复制初始目标，不做任何筛选

        
        # blocked_goal_names = [bg.get("goal_name", "") for bg in self.blocked_goals_round]
        goal_comprehensive_scores = {}

        # 新目标综合得分计算

        for goal in self.goal_list:
            goal_name = goal.get("name", "")
            if not goal_name:
                continue
            # 过滤终阻/抛弃目标
            if goal_name in self.secondary_blocked_goals or goal_name in self.discarded_goals:
                goal_comprehensive_scores[goal_name] = -100
                continue
            importance_score = goal.get("score", 0.0)
            latest_progress = 0.0

            if self.infs != 1:
                try:
                    round_items = list(self.goal_progress_record.get(goal_name, {}).items())
                except Exception as e:
                    round_items = []
                if round_items:
                    try:
                        round_items.sort(key=lambda x: int(x[0].replace("Round ", "")))
                        latest_progress = round_items[-1][1][0]
                    except (ValueError, IndexError):
                        latest_progress = 0.0

            comprehensive_score = (importance_score - latest_progress / 2)
            goal_comprehensive_scores[goal_name] = comprehensive_score

        # 新目标排序
        self.goal_list.sort(key=lambda g: goal_comprehensive_scores.get(g.get("name", ""), -1), reverse=True)
        self.goal_list = [g for g in self.goal_list if goal_comprehensive_scores.get(g.get("name",""), -1) >= 0]
        print(f"aaa排序的self.goal_list有{len(self.goal_list)}，是：{self.goal_list}")

        # 新目标筛选 - 过滤原始目标（避免重复）
        original_goal_names = [g.get("name", "") for g in original_goals_selected]
        
        filtered_goals_candidates = [g for g in self.goal_list if g.get("name", "") not in original_goal_names]
        print(f"self.goal_list筛除初始的之后有{len(filtered_goals_candidates)}，是：{filtered_goals_candidates}")

        # 新目标筛选 - 递进目标优先
        filtered_goals_selected = []
        prog_goals = []


        for goal in filtered_goals_candidates:
            goal_name = goal.get("name", "")
            if goal_name in self.discarded_goals:
                continue

            if goal['is_progressive'] == True:
                prog_goals.append(goal)

                
        # 新目标数量限制
        if len(prog_goals) >= self.top_k:
            filtered_goals_selected = prog_goals[:self.top_k]
            print(f"递进目标全了：{filtered_goals_selected}")
        else:
            filtered_goals_selected.extend(prog_goals)
            need_num = self.top_k - len(filtered_goals_selected)
            print(f"递进目标不全：差{need_num}个\n")
            print(f"递进目标现在是：{filtered_goals_selected}")
            for goal in filtered_goals_candidates:
                if need_num<=0:
                    break
                goal_name = goal.get("name", "")
                if (goal not in filtered_goals_selected and goal_name not in self.discarded_goals):
                    filtered_goals_selected.append(goal)
                    need_num -= 1
            print(f"新目标补全后是：{filtered_goals_selected}")
        

        print(f"原始目标列表是：{original_goals_selected}")
        
        selected_goals = original_goals_selected + filtered_goals_selected
        self._log_info(f"目标排序完成 | 原始目标数: {len(original_goals_selected)} | 新目标数: {len(filtered_goals_selected)} | 总计: {len(selected_goals)}")
        return selected_goals

    def eval_monitor(self, round_num, top_k_goals, history_dialogs, current_round_dialog, combined_result):
        self._log_info("开始目标进度监控评估")
        round_name = f"Round {round_num}"
        top_k_goal_names = [g.get("name", "") for g in top_k_goals if g.get("name")]
        all_goal_names = [g.get("name", "") for g in self.goal_list if g.get("name")]

        top_k_goal_set_str = []
        for idx, g in enumerate(top_k_goals):
            g_name = g.get("name", f"Goal {idx+1}")
            g_desc = g.get("description", "No description")
            top_k_goal_set_str.append(f"{idx+1}. Name: {g_name}\nDescription: {g_desc}")
        top_k_goal_set_str = "\n\n".join(top_k_goal_set_str)

        history_dialogs_str = self.process_history_dialogs(history_dialogs)
        current_round_dialog_str = f"Patient: {current_round_dialog.get('patient', '')}\nDoctor: {current_round_dialog.get('doctor', '')}"

        patient_answer = current_round_dialog.get('patient', '').lower()
        

        monitor_prompt_template = """
You are a senior clinical assessment expert. Evaluate the completion progress of multiple goals in the current consultation round solely based on the full doctor-patient dialogue history and latest round dialogue. Follow rules strictly.

## Input Information
1. Current Round: {round_num}
2. Goals to Evaluate This Round:
{top_k_goal_set_str}
3. Full Dialogue History:
{history_dialogs_str}
4. Latest Round Dialogue:
{current_round_dialog_str}

## Evaluation Rules (Mandatory)
1. Progress Value (0.0-1.0, 1 decimal place; determined solely by dialogue relevance to the goal and completion proximity):
   - 0.0: No dialogue related to the goal; no progress made
   - 0.1-0.4: Minimal weak-relevant dialogue; low progress
   - 0.5-0.9: Abundant relevant dialogue; high progress (higher score = closer to completion)
   - 1.0: Fully meets goal requirements; completely accomplished

2. Progress Explanation: 1-2 specific sentences, justifying the value with dialogue content; no vague descriptions.
3. Coverage: Evaluate all goals; no omissions.

## Output Format: Strict JSON only (no extra content); key = goal name.
{{
  "Goal Name 1": {{
    "progress_value": 0.8,
    "progress_explain": "Specific justification"
  }}
}}

## Forbidden: No content other than JSON.
"""
        monitor_prompt = monitor_prompt_template.format(
            round_num=round_num, 
            top_k_goal_set_str=top_k_goal_set_str,
            history_dialogs_str=history_dialogs_str, 
            current_round_dialog_str=current_round_dialog_str
        )
        
        try:
            monitor_response = self.monitor.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[{"role": "user", "content": monitor_prompt}], 
                response_format={"type": "json_object"},
                timeout=60
            )
            eval_raw = monitor_response.choices[0].message.content.strip()
            tokens_cost = monitor_response.usage.total_tokens
            #r 和目标管理有关，因为这个就是完全的目标进度评估器
            self.sys_tokens_cost += tokens_cost
            self.goal_tokens_cost += tokens_cost

        except Exception as e:
            self._log_info(f"监控LLM调用失败: {e}", "error")
            eval_raw = "{}"

        for goal_name in all_goal_names:
            if not goal_name:
                continue
            is_top_k = goal_name in top_k_goal_names
            progress_explain = ""
            goal = next((g for g in self.goal_list if g.get('name') == goal_name), None)
            if not goal:
                status = "normal"
            else:
                is_progressive = goal.get('is_progressive', False)
                if is_top_k:
                    status = "secondary_blocked" if is_progressive else "primary_blocked"
                else:
                    status = "normal"

            if goal_name in self.goal_progress_record:
                try:
                    last_round_name = list(self.goal_progress_record[goal_name].keys())[-1]
                    progress_value = self.goal_progress_record[goal_name][last_round_name][0]
                except (IndexError, KeyError):
                    progress_value = 0.0
                self.goal_progress_record[goal_name][round_name] = (progress_value, is_top_k, progress_explain, status)
            else:
                progress_value = 0.0
                self.goal_progress_record[goal_name] = {round_name: (progress_value, is_top_k, progress_explain, status)}

        top_k_progress_dict = {}
        try:
            eval_dict = json.loads(eval_raw)
            for goal_name in top_k_goal_names:
                matched = False
                for eval_key in eval_dict.keys():
                    if goal_name.lower().strip() == eval_key.lower().strip():
                        goal_eval = eval_dict[eval_key]
                        progress_value = float(goal_eval.get("progress_value", 0.0))
                        progress_value = max(0.0, min(1.0, round(progress_value,1)))
                        progress_explain = goal_eval.get("progress_explain", "")
                        top_k_progress_dict[goal_name] = (progress_value, progress_explain)
                        matched = True
                        break
                if not matched:
                    top_k_progress_dict[goal_name] = (0.0, f"Round {round_num}: No matching evaluation data")
        except json.JSONDecodeError as e:
            self._log_info(f"监控JSON解析失败: {e} | 原始输出: {eval_raw[:100]}", "error")
            for goal_name in top_k_goal_names:
                top_k_progress_dict[goal_name] = (0.0, "JSON parse error")
        except Exception as e:
            self._log_info(f"监控评估其他错误: {e}", "error")
            for goal_name in top_k_goal_names:
                top_k_progress_dict[goal_name] = (0.0, f"Evaluation error: {str(e)[:50]}")

        for goal_name in top_k_goal_names:
            if not goal_name:
                continue
            progress_value, progress_explain = top_k_progress_dict.get(goal_name, (0.0, "No data"))
            current_status = self.goal_progress_record[goal_name][round_name][3]
            self.goal_progress_record[goal_name][round_name] = (progress_value, True, progress_explain, current_status)
            self._log_info(f"目标进度更新 | {goal_name}: {progress_value} | 状态: {current_status}")

        self._log_info(f"监控评估完成 | 本轮评估目标数: {len(top_k_goal_names)} | 总进度记录数: {len(self.goal_progress_record)}")

    def create_filter_goals_one_llm(self, history_dialogs: List[Dict]) -> tuple[List[tuple], List[Dict]]:
        self._log_info("开始生成递进/对立/互补目标")
        self.edges.clear()
        valid_goals = []
        recent_history = self.process_history_dialogs(history_dialogs, -2)
        history_dialogs_str = self.process_history_dialogs(history_dialogs)
        blocked_goals_list = self.blocked_goals()
        self.blocked_goals_round = blocked_goals_list

        blocked_goals_str = "NO BLOCKED GOALS - No need to generate progressive goals" if not blocked_goals_list else "\n".join([
            f"{idx+1}. {bg['goal_name']} (Progress Increase: {bg['progress_increase']:.2f}, Latest 2 Rounds Progress: {bg[list(bg.keys())[1]]['progress']:.2f}/{bg[list(bg.keys())[2]]['progress']:.2f})" 
            for idx, bg in enumerate(blocked_goals_list)
        ])

        max_display_goals = 6
        display_goal_list = self.goal_list[:max_display_goals] if len(self.goal_list) > max_display_goals else self.goal_list
        goal_trunc_note = f"\n(Note: Only top {len(display_goal_list)} core goals are displayed for focused generation)" if len(self.goal_list) > max_display_goals else ""
        existing_goals_str = "NO EXISTING CORE GOALS - Generate basic opposing/complementary goals for general consultation" if not self.goal_list else "\n".join([
            f"{idx+1}. Name: {g['name']}\nDescription: {g['description']}" for idx, g in enumerate(display_goal_list)
        ]) + goal_trunc_note

        existing_goal_names = [g.get("name") for g in self.original_goals_round if g.get("name")]
        existing_goals_prompt = ", ".join(existing_goal_names) if existing_goal_names else "None"
        discarded_goals_prompt = ", ".join(self.discarded_goals) if self.discarded_goals else "None"

        # ===== 新增：概率触发PubMed知识参考 =====
        pubmed_knowledge = ""
        # 生成随机数判断是否触发（0-1之间）
        random_val = self.random.random()
        if random_val <= self.pubmed_reference_prob:
            self._log_info(f"触发PubMed知识参考（随机值={random_val:.2f} ≤ 概率={self.pubmed_reference_prob:.2f}）")
            # 步骤1：生成PubMed搜索词（基于对话历史+受阻目标）
            search_prompt = [
                {
                    "role": "user",
                    "content": f"""
                    Based on the following doctor-patient dialogue history and blocked consultation goals, generate 2-3 concise search phrases for PubMed (medical literature database), focusing on clinical diagnosis/treatment related to the patient's symptoms.
                    Strictly output JSON format only: {{"search_phrases": ["phrase1", "phrase2", ...]}}

                    Dialogue History:
                    {history_dialogs_str[:500]}  

                    Blocked Goals (if any):
                    {blocked_goals_str[:300]}
                    """
                }
            ]
            # 调用rag_handler生成搜索词
            search_list, tokens_cost_pubmed = self.rag_handler.gen_search_list(search_prompt)

            #r 查询结果为生成目标服务
            self.sys_tokens_cost += tokens_cost_pubmed
            self.goal_tokens_cost += tokens_cost_pubmed


            # 调用rag_handler获取PubMed文献摘要
            #print(f"检索关键词：\n{search_list}\n")
            pubmed_results = self.rag_handler.get_pubmed_results(search_list)

            # 格式化PubMed结果（仅保留有效内容，截断控制长度）
            a = pubmed_results.get("searchs")
            #print(f"条件1：\n{a}\n")
            if pubmed_results.get("searchs") and pubmed_results["searchs"] != "No search results":
                pubmed_items = []
                for item in pubmed_results["searchs"]:
                    if isinstance(item, dict) and item.get("content") and item["content"] != "Error in retrieving content":
                        # 截断摘要到300字符，保留核心信息
                        content = item["content"][:300].strip()
                        pubmed_items.append(f"Search Phrase: {item['search_item']}\nRelevant Medical Knowledge: {content}")
                if pubmed_items:
                    pubmed_knowledge = "\n\n".join(pubmed_items)
                    self._log_info(f"获取到PubMed知识，共{len(pubmed_items)}条相关文献摘要")
                else:
                    pubmed_knowledge = "NO VALID PUBMED KNOWLEDGE FOUND"
            else:
                pubmed_knowledge = "NO PUBMED SEARCH RESULTS"
        else:
            self._log_info(f"未触发PubMed知识参考（随机值={random_val:.2f} > 概率={self.pubmed_reference_prob:.2f}）")

        # ===== 原有Prompt基础上融入PubMed知识 =====
        
        print(f"外部知识：\n{pubmed_knowledge}\n")
        unified_prompt = f"""
    You are an expert in clinical decision-making and doctor-patient communication, specializing in generating consultation goals to guide inquiry strategies.

    # External Medical Knowledge (from PubMed, for reference only)
    {pubmed_knowledge}

    ## Core Constraint (HIGHEST PRIORITY)
    Goals are strictly divided into 3 levels; no further detailing is allowed.
    1. Level 1 (Macro Original): Same breadth as core goals (diagnostic accuracy, safety, consultation comprehensiveness, patient emotional care), purely high-level, no specific body part or symptom.
    2. Level 2 (Slightly Detailed): Focus on inquiry methods including communication refinement, questioning logic, consultation strategy, and diagnostic reasoning.
    3. Level 3 (Specific Focus): Targeted investigation of specific anatomical regions or clinical symptom clusters to advance differential diagnosis;
        [RED LINE] Goal names MUST follow: "Focus on patient XX", "Screen patient XX", "Evaluate patient XX";
        [PROHIBITION] Strictly forbidden to reduce the goal to a single data point, binary judgment, or a specific question.

    ## Generation Basis
    1. Blocked goal list (may be empty). Progressive goals are generated ONLY for these goals:
    {blocked_goals_str}

    2. Existing core goals, used as the **common basis** for generating opposing goals and complementary goals (These are NOT the same group of goals, but different categories of goals):
    {existing_goals_str}

    3. Historical doctor-patient dialogue:
    {history_dialogs_str}

    4. Latest two rounds of dialogue, used for the analysis of blocked goals (if any):
    {recent_history}

    ## Output Format Rules (Violation = Invalid Output)
    1. Goal name: ≤ 6 words.
    2. Goal description: ≤ 20 words, explaining the implementation logic and consultation value.
    3. Score: Only plain digits 1/2/3 are used (scores assigned by clinical importance: 1=Auxiliary, 2=Core, 3=Critical), with no symbols of any kind.
    4. Relationships are only shown in tuples; the "bind_to" field is forbidden in the goal content.
    5. No redundant explanations, unified format, each goal block starts with *The Xth*.
    6. Abandoned duplicate/similar goals: do not appear, are not numbered, and leave no traces.
    7. Each goal block must include a level field (level 1 / level 2/ level 3), You must consider the necessity of generating Level 3 goals.

    ## [Mandatory Fixed Generation Steps] Must be executed in the following 5-step order, no reversal, no omission
    Step 1: Follow the rules below:
        - If there ARE blocked goals: Generate goal names **one by one** in the fixed sequence: **Progressive Goals → Opposing Goals → Complementary Goals**. Generate only one name at a time, with NO description, score, or relationship.
        - If there are NO blocked goals: Generate goal names **one by one** in the fixed sequence: **Opposing Goals → Complementary Goals**. Generate only one name at a time, with NO description, score, or relationship.

    Step 2: Compare the newly generated name one by one with the goal names in the following two sets:
        - List of existing core goal names: {existing_goals_prompt}
        - List of discarded goal names: {discarded_goals_prompt}

    Step 3: Similarity judgment rules (any one met = judged as "too similar"):
        1) The names are almost identical;
        2) The semantics are basically the same (e.g., Accuracy and Diagnostic Accuracy);
        3) The described objects are the same, and the core themes completely overlap;
        4) Only synonym replacement or paraphrasing.

    Step 4: Follow the rules below and count the number of qualified goals simultaneously:
        - Qualified name (not similar, not duplicated): Keep the name, supplement description, score, and relationship, then proceed to generate the next one; and add 1 to the number of qualified goals in the current category.
        - Unqualified name (duplicated/similar): Discard it immediately and completely, generate no follow-up content, **return directly to Step 1 to generate a replacement name**, no skipping; the number of qualified goals in the current category will NOT be increased.

    Step 5: Verify the quantity category by category in sequence, then verify the total quantity after all categories are qualified, according to the following rules:
        1) First generate progressive goals (execute this item ONLY if there are blocked goals; ignore if no blocked goals): The number of qualified goals must be strictly equal to the total number of blocked goals {len(blocked_goals_list)}, with a one-to-one correspondence and binding in order; only after the quantity of progressive goals meets the standard can the generation of opposing goals start.
        2) The number of qualified opposing goals must be strictly equal to {self.k_opp_comp}; only after meeting the standard can the generation of complementary goals start.
        3) The number of qualified complementary goals must be strictly equal to {self.k_opp_comp}; only after meeting the standard can the total qualified quantity verification be performed.
        4) Total qualified quantity verification: If the total quantity of the three types of goals = {len(blocked_goals_list)+self.k_opp_comp*2}, stop the entire generation process.

    ## Required Output Format (NO DEVIATION ALLOWED), example as follows:
    ### Progressive Goals (prog)
    ... (Intermediate goals are generated following the same rules, DO NOT omit in actual generation)
    *The 1th*
    name: (the name of the goal)
    level: (the level of the goal)
    description: (the description of the goal)
    score: (the score of the goal)

    ... (Repeat for all {len(blocked_goals_list)} goals)

    ### Opposing Goals (opp)
    ... (Intermediate goals are generated following the same rules, DO NOT omit in actual generation)

    ### Complementary Goals (comp)
    ... (Intermediate goals are generated following the same rules, DO NOT omit in actual generation)

    ### Relationship Tuples (one per line, full coverage with no omissions) (Format: (NewGoal, LinkedGoal, Type) | Type: prog/opp/comp(MUST be chosen from the three options, NOT unknown))
    (Strategic Comprehensiveness, Improve Consultation Efficiency, opp)
    """



        try:
            llm_output, tokens_cost = query_model_ours(self.fuzhu_llm, unified_prompt, "")
            #r token计入
            self.sys_tokens_cost += tokens_cost
            self.goal_tokens_cost += tokens_cost
        except Exception as e:
            self._log_info(f"目标生成LLM调用失败: {e}", "error")
            llm_output = ""
        self._log_info(f"第{self.infs}轮，目标生成LLM输出：{llm_output}")

        prog_pattern = re.compile(
            r'###\s*PROGRESSIVE\s+GOALS\s*\(prog\)(.*?)###\s*OPPOSING\s+GOALS\s*\(opp\)',
            re.DOTALL | re.IGNORECASE
        )
        
        opp_pattern = re.compile(
            r'###\s*OPPOSING\s+GOALS\s*\(opp\)(.*?)###\s*COMPLEMENTARY\s+GOALS\s*\(comp\)',
            re.DOTALL | re.IGNORECASE
        )
        
        comp_pattern = re.compile(
            r'###\s*COMPLEMENTARY\s+GOALS\s*\(comp\)(.*?)###\s*RELATIONSHIP\s+TUPLES',
            re.DOTALL | re.IGNORECASE
        )
        
        tuple_pattern = re.compile(
            r'\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*(prog|opp|comp)\s*\)',
            re.DOTALL | re.IGNORECASE
        )
        
        goal_block_pattern = re.compile(
            r'\*The\s+\d+(?:st|nd|rd|th)\*\s*(.*?)(?=\*The|###|$)',
            re.DOTALL | re.IGNORECASE
        )

        prog_content = ""
        if prog_pattern.search(llm_output):
            prog_content = prog_pattern.search(llm_output).group(1).strip()

        opp_content = ""
        if opp_pattern.search(llm_output):
            opp_content = opp_pattern.search(llm_output).group(1).strip()

       
        comp_content = ""
        if comp_pattern.search(llm_output):
            comp_content = comp_pattern.search(llm_output).group(1).strip()

        relation_tuples = tuple_pattern.findall(llm_output)

        # 解析各类目标（逻辑不变，但因为comp_content有值，comp_goals不再为空）
        prog_goals = []
        for block in goal_block_pattern.findall(prog_content):
            if block.strip():
                goal = self.parse_goal(block)
                goal['is_progressive'] = True
                prog_goals.append(goal)

        opp_goals = []
        for block in goal_block_pattern.findall(opp_content):
            if block.strip():
                opp_goals.append(self.parse_goal(block))

        empty_bound_opp_goal_names = set()
        for new_goal_name, bound_goal_name, rel_type in relation_tuples:
            # 清理名称中的空格、引号，统一格式
            clean_new_name = new_goal_name.strip().strip('"').strip("'").lower()
            clean_bound_name = bound_goal_name.strip().strip('"').strip("'")
            # 筛选条件：对立目标(opp) + 关联目标为空
            if rel_type.lower() == "opp" and not clean_bound_name:
                empty_bound_opp_goal_names.add(clean_new_name)

        # 过滤掉关联目标为空的对立目标
        opp_goals = [
            goal for goal in opp_goals 
            if goal.get("name", "").strip().lower() not in empty_bound_opp_goal_names
        ]

        
        comp_goals = []
        for block in goal_block_pattern.findall(comp_content):
            if block.strip():
                comp_goals.append(self.parse_goal(block))

        all_generated_goals = prog_goals + opp_goals + comp_goals

        print(f"第{self.infs}轮，所有提取的目标信息：{all_generated_goals}")

        original_names = {
            og.get("name","").lower().strip()
            for og in self.original_goals_round
            if isinstance(og, dict) and og.get("name")
        }

        unique_goals = []
        seen_names = set()
        for g in all_generated_goals:
            name = g.get("name","").lower().strip()
            if name and name not in original_names and name not in seen_names and name not in [n.lower() for n in self.discarded_goals]:
                seen_names.add(name)
                unique_goals.append(g)
        all_generated_goals = unique_goals
        print(f"第{self.infs}轮，去重后的目标信息：{all_generated_goals}")
        self._log_info(f"目标去重完成 | 原始生成: {len(prog_goals)+len(opp_goals)+len(comp_goals)} | 去重后: {len(all_generated_goals)}")

        for new_goal_name, bound_goal_name, rel_type in relation_tuples:
            new_goal_name = new_goal_name.strip().strip('"').strip("'")
            bound_goal_name = bound_goal_name.strip().strip('"').strip("'")
            if rel_type in ["prog", "opp", "comp"] and new_goal_name:
                relation_tuple = (new_goal_name, bound_goal_name, rel_type)
                print(f"第{self.infs}轮，添加进的关系：{relation_tuple}")
                self.edges.append(relation_tuple)

        valid_goals = [g for g in all_generated_goals if g.get("score", 0.0) >= self.score_threshold]

        self.goal_list.extend(valid_goals)
        self._init_missing_goal_progress()

        self._log_info(f"目标生成完成 | 递进: {len(prog_goals)} | 对立: {len(opp_goals)} | 互补: {len(comp_goals)} | 有效目标: {len(valid_goals)}")
        return self.edges, valid_goals





    def top_k_reason_and_combine_one_llm(self, top_k_goals: List[Dict], previous_dialogue: str = "", use_cot: bool = True) -> str:
        if not top_k_goals:
            self._log_info("无有效Top-K目标，跳过COT推理", "warning")
            return "No valid top-k goals for reasoning"

        self._log_info(f"开始COT推理 | 推理目标数: {len(top_k_goals)}")
        goal_info_list = []
        for idx, goal in enumerate(top_k_goals):
            goal_name = goal.get('name', f"Goal {idx+1}")
            goal_desc = goal.get('description', "No description")
            rel_type = None
            related_goal = ""
            for edge in self.edges:
                if len(edge)>=3 and edge[0].strip().lower() == goal_name.lower():
                    rel_type = edge[2]
                    related_goal = edge[1] if edge[1] else ""
                    break
            goal_info_list.append({
                "index": idx + 1,
                "name": goal_name,
                "description": goal_desc,
                "rel_type": rel_type or "none",
                "related_goal": related_goal,
                "is_progressive": goal.get('is_progressive', False)
            })

        unified_prompt = f"""
You are a Senior Clinical Reasoning Expert, specialized in doctor-patient consultation goal reasoning. Complete TWO tasks in one output (strict format).

## CLINICAL COT REASONING GUIDANCE (MANDATORY)
You MUST learn the intrinsic logic and rules from the following examples, draw inferences from one instance, and adapt the reasoning to the actual dialogue context.

## CLINICAL COT REASONING EXAMPLES (FOLLOW THIS LOGIC FOR ALL GOALS)
### Example 1: Progressive Goal (prog) - Replace Original Inquiry Method (bind to blocked core goal)
1. Dialogue History: Original inquiry method fails to obtain information, core goal progress is blocked.
2. Goal Value: Change inquiry perspective/expression to re-attempt information collection for core goal.
3. Inquiry Action: Replace complex/abstract questions with simple/concrete expressions, adjust inquiry angle.
4. Clinical Conclusion: Prioritize changing inquiry strategy to break through information collection bottleneck.

### Example 2: Core Goal (non-prog) - Direct Information Collection
1. Dialogue History: No relevant information collected yet, core goal progress is 0.
2. Goal Value: Collect key information to promote core goal advancement.
3. Inquiry Action: Ask targeted questions based on goal requirements, ensure clear and concise expression.
4. Clinical Conclusion: Prioritize direct information collection to lay foundation for diagnosis.

## TASK 1: Chain-of-Thought (COT) Reasoning for EACH Top-K Goal
Follow the clinical example logic (4 steps: Dialogue History → Goal Value → Inquiry Action → Clinical Conclusion).
Requirements: 1. Step-by-step, clinically accurate; 2. Concise, no redundancy; 3. Combine dialogue history & goal info; 4. No micro-goals.

## TASK 2: Integrate COT Results into a Coherent Summary
Accurately reflect goal relationships:
1. Progressive (prog): Change inquiry strategy for blocked core goals (only one layer, no new progressive goals);
2. Opposing (opp): Clinical trade-off considerations for comprehensive consultation;
3. Complementary (comp): Supplementary dimensions for holistic clinical inquiry;
4. No headings/lists, only one continuous paragraph, retain core reasoning steps.

## INPUT INFORMATION
1. Doctor-Patient Dialogue History:
{previous_dialogue}
2. Top-K Consultation Goals (generate COT for each):
{json.dumps(goal_info_list, indent=2, ensure_ascii=False)}

## STRICT OUTPUT FORMAT (NO EXTRA TEXT/EXPLANATION)
### COT REASONING FOR EACH GOAL
Goal 1: [4-step COT reasoning result]
Goal 2: [4-step COT reasoning result]
... (one line per goal, match top-k number)

### COMBINED SUMMARY
[Continuous integration paragraph, follow relationship rules]
"""
        try:
            llm_output, tokens_cost = query_model_ours(self.fuzhu_llm, unified_prompt, "")
            #r token计入
            self.sys_tokens_cost += tokens_cost
            self.goal_tokens_cost += tokens_cost
        except Exception as e:
            self._log_info(f"COT推理LLM调用失败: {e}", "error")
            llm_output = "### COT REASONING FOR EACH GOAL\nGoal 1: No reasoning due to LLM error\n### COMBINED SUMMARY\nNo valid reasoning result"
        self._log_info(f"COT推理LLM输出（前300字）：{llm_output[:300]}")

        summary_pattern = re.compile(r'###\s*COMBINED\s+SUMMARY\s*\n(.*)', re.DOTALL | re.IGNORECASE)
        cot_pattern = re.compile(r'###\s*COT\s+REASONING\s+FOR\s+EACH\s+GOAL\s*\n(.*?)###\s*COMBINED\s+SUMMARY', re.DOTALL | re.IGNORECASE)

        combined_result = summary_pattern.search(llm_output).group(1).strip() if (summary_pattern.search(llm_output) and summary_pattern.search(llm_output).group(1)) else "No valid reasoning result"
        cot_content = cot_pattern.search(llm_output).group(1).strip() if (cot_pattern.search(llm_output) and cot_pattern.search(llm_output).group(1)) else ""

        self.top_k_results = []
        if cot_content:
            for line in cot_content.split("\n"):
                line_stripped = line.strip()
                if line_stripped.startswith("Goal "):
                    cot_result = line_stripped.split(":", 1)[1].strip() if ":" in line_stripped else ""
                    self.top_k_results.append(cot_result)
        if len(self.top_k_results) < len(top_k_goals):
            self.top_k_results += ["No COT reasoning for this goal"] * (len(top_k_goals) - len(self.top_k_results))

        self._log_info("COT推理完成，生成整合结论")
        return combined_result

    def reason_round_one_llm(self, history_dialogs: List[Dict]) -> str:
        self._log_round_start()
        
        if self.infs == 1:
            #r 病例间不影响，每次第一轮置空token计数
            self.token_cost = 0
            self.sys_tokens_cost = 0
            self.goal_tokens_cost = 0


            self.goal_list = self.rag_handler.goals.get('nodes', [])
            self.initial_goal_list = self.goal_list.copy()
            self._init_missing_goal_progress()
            self._log_info(f"初始化核心目标 | 数量: {len(self.goal_list)}")
            

        self.original_goals_round = self.goal_list.copy()
        _, deduped_goals = self.create_filter_goals_one_llm(history_dialogs)
        self._init_missing_goal_progress()
        self.filtered_goals_round = deduped_goals

        top_k_goals = self.sort_goals()
        self.top_k_goals_round = top_k_goals
        self.top_k_goals[f"round_{self.infs}"] = self.top_k_goals_round

        # 新增：推理时调用RAG检索经验
        query_dict = {
            "current_goals": [g.get("name") for g in top_k_goals],
            "previous_dialogue": self.process_history_dialogs(history_dialogs)
        }
        rag_reasoning_results = self.retrieve_for_reasoning(query_dict)
        
        rag_context = "\n".join([f"Relevant experience for {goal}: {doc.page_content[:200]}..." for goal, docs in rag_reasoning_results.items() for doc in docs])
        previous_dialogue = self.process_history_dialogs(history_dialogs) + "\n\nRelevant clinical experience:\n" + rag_context

        combined_result = self.top_k_reason_and_combine_one_llm(top_k_goals, previous_dialogue, use_cot=True)
        self.combine_results_record[self.infs] = combined_result

        self._log_round_end()
        
        print(f"------------------------第{self.infs}轮----------------------")
        generated_goals_with_type = []
        for edge in self.edges:
            goal_name, bound_name, rel_type = edge
            goal = next((g for g in self.filtered_goals_round if g.get('name', '').lower() == goal_name.lower()), None)
            if goal:
                generated_goals_with_type.append({
                    'name': goal_name,
                    'type': rel_type,
                    'description': goal.get('description', ''),
                    'score': goal.get('score', 0.0)
                })
        for goal in self.filtered_goals_round:
            goal_name = goal.get('name', '')
            if not any(g['name'].lower() == goal_name.lower() for g in generated_goals_with_type):
                rel_type = 'unknown'
                for edge in self.edges:
                    if edge[0].lower() == goal_name.lower():
                        rel_type = edge[2]
                        break
                generated_goals_with_type.append({
                    'name': goal_name,
                    'type': rel_type,
                    'description': goal.get('description', ''),
                    'score': goal.get('score', 0.0)
                })
        print("1. 本轮产生的目标列表及类型：")
        if generated_goals_with_type:
            for idx, g in enumerate(generated_goals_with_type, 1):
                print(f"   {idx}. 名称: {g['name']} | 类型: {g['type']} | 描述: {g['description']} | 分数: {g['score']}")
        else:
            print("   无")
        
        print("2. 本轮筛选后的目标列表（Top-K）：")
        if self.top_k_goals_round:
            for idx, g in enumerate(self.top_k_goals_round, 1):
                print(f"   {idx}. 名称: {g.get('name', 'Unnamed')} | 描述: {g.get('description', 'No description')} | 分数: {g.get('score', 0.0)}")
        else:
            print("   无")
        
        current_round_discarded = [g for g in self.discarded_goals if g not in sum(self.round_discarded_goals.values(), [])]
        self.round_discarded_goals[self.infs] = current_round_discarded
        print("3. 本轮抛弃的目标列表：")
        if current_round_discarded:
            for idx, g in enumerate(current_round_discarded, 1):
                print(f"   {idx}. {g}")
        else:
            print("   无")
        print("--------------------------------------------")
        
        return combined_result

    import re  # 确保文件顶部导入了re

    def inference_doctor(self, question) -> str:
        if not self.history_dialogue:
            self.history_dialogue = []

        history_dialogue = self.process_history_dialogs(self.history_dialogue)
        self.reason_round_one_llm(self.history_dialogue)

        is_finish = self.is_finish_prompt if (self.infs > 1 and self.is_finish_doctor()) else ''

        # ===================== 单轮检索 + 精准提取纯对话（舍弃推理/评估） =====================
        rag_response_docs = self.retrieve_for_response(history_dialogue)


        #print(f"回复时检索内容：{rag_response_docs}")
        rag_response_context = ""

        if rag_response_docs:
            # 1. 强过滤：只保留同场景的单轮经验
            same_scene_single_round_docs = [
                doc for doc in rag_response_docs
                if doc.metadata.get("bench_name", "").strip() == self.scenario.strip()
            ]

            self._log_info(
                f"回复检索完成：原始返回{len(rag_response_docs)}条单轮经验，"
                f"严格同场景过滤后保留{len(same_scene_single_round_docs)}条单轮经验",
                "info"
            )

            # 2. 对每条单轮经验，精准提取「纯对话部分」（舍弃推理/评估）
            exp_pieces = []
            # 正则匹配：提取 "Conversation in the current round X" 后的纯对话内容
            conv_pattern = re.compile(
                r'Conversation in the current round \d+:\s*(.*?)(?=\nDoctor\'s reasoning outcome|Evaluation in the current round|$)',
                re.DOTALL | re.IGNORECASE
            )

            for idx, doc in enumerate(same_scene_single_round_docs):
                single_round_text = doc.page_content.strip()
                # 精准匹配纯对话部分
                conv_match = conv_pattern.search(single_round_text)
                if conv_match:
                    # 提取并清理纯对话内容（去掉多余空格/换行）
                    pure_conversation = conv_match.group(1).strip()
                    # 只保留对话，舍弃推理/评估，自然控制长度，无需盲截断
                    if pure_conversation:
                        exp_pieces.append(
                            f"=== Historical Single Round Conversation {idx+1} ===\n{pure_conversation}"
                        )
                else:
                    self._log_info(f"第{idx+1}条单轮经验未匹配到纯对话内容，跳过", "warning")

            # 拼接最终的参考上下文（只有纯对话，无冗余）
            rag_response_context = "\n\n".join(exp_pieces) if exp_pieces else "No relevant single-round conversation reference"
        else:
            self._log_info("回复检索：未匹配到任何同场景单轮经验", "info")
        # ========================================================================================

        doctor_respond_prompt_template = PromptTemplate(
                template="""
Current round: {round_num}
Generate a response to the patient based on the given context:

1. Dialogue history (important decision basis):
{history_dialogue}

2. Patient's latest utterances:
{latest_response_patient}

3. Current round reasoning result (core decision basis):
{combined_result}

4. Referenced historical single-round conversations (same scenario only, for reference only, pure doctor-patient dialogue without reasoning/evaluation):
{rag_response_context}

{is_finish}
    """,
            input_variables=["round_num", "history_dialogue", "latest_response_patient",
                             "combined_result", "rag_response_context", "is_finish"]
        )

        combined_result = self.combine_results_record.get(self.infs, "No reasoning result")

        doctor_respond_prompt = doctor_respond_prompt_template.format(
            round_num=self.infs,
            history_dialogue=history_dialogue,
            latest_response_patient=question,
            combined_result=combined_result,
            rag_response_context=rag_response_context,
            is_finish=is_finish
        )

        self._log_info("Calling LLM to generate current round reply")
        answer, tokens = query_model(self.backend, doctor_respond_prompt, self.init_doctor_prompt, self.base_url, self.api_key)
        self.token_cost += tokens
        #r 不算目标管理
        self.sys_tokens_cost += tokens
        self._log_info(f"Reply generated | tokens this round: {tokens} | total tokens: {self.token_cost}")

        self.history_dialogue.append({'patient': question, 'doctor': answer})

        try:
            self.eval_monitor(
                self.infs, self.top_k_goals_round, self.history_dialogue,
                {'patient': question, 'doctor': answer}, combined_result
            )
        except Exception as e:
            self._log_info(f"Round evaluation failed: {e}", "error")

        parsed_output = {}
        for line in answer.strip().split('\n'):
            if ": " in line:
                key, value = line.split(": ", 1)
                parsed_output[key.strip().lower()] = value.strip()
        true_answer = parsed_output.get("response", answer)

        if parsed_output.get("action") == "diagnose":
            self._log_info(f"本轮ACTION=diagnose，结束问诊 | 诊断结果: {parsed_output.get('diagnosis', '无')}", "info")
            # 打印问诊结束指定信息
            print("++++++++++++++++问诊结束打印++++++++++++++++++")
            print("1. 问诊最开始的目标列表：")
            if self.initial_goal_list:
                for idx, g in enumerate(self.initial_goal_list, 1):
                    print(f"   {idx}. 名称: {g.get('name', 'Unnamed')} | 描述: {g.get('description', 'No description')} | 分数: {g.get('score', 0.0)}")
            else:
                print("   无")
            print("2. 问诊结束后的目标列表：")
            if self.goal_list:
                for idx, g in enumerate(self.goal_list, 1):
                    print(f"   {idx}. 名称: {g.get('name', 'Unnamed')} | 描述: {g.get('description', 'No description')} | 分数: {g.get('score', 0.0)}")
            else:
                print("   无")
            print("3. 抛弃的目标列表：")
            if self.discarded_goals:
                for idx, g in enumerate(self.discarded_goals, 1):
                    print(f"   {idx}. {g}")
            else:
                print("   无")
            print("===============================")

            # 问诊结束 → 执行目标集更新 + 经验筛选入库
            # self.updated_goals()
            self.filter_update_exps(bench_name=self.scenario, patient_id=self.case_id)

            self._log_case_end()  # 病例结束

        self.infs += 1
        return answer
    def respond(self, message: Dict, **kwargs) -> str:
        patient_msg = message.get("patient", "")
        return self.inference_doctor(patient_msg)

    def is_finish_doctor(self) -> bool:
        original_goals = self.rag_handler.goals.get('nodes', [])
        if not original_goals:
            return False
        
        for goal in original_goals:
            if isinstance(goal, dict):
                goal_name = goal.get('name', '')
            elif isinstance(goal, str):
                goal_name = goal
            else:
                goal_name = ""
            
            if not goal_name or goal_name not in self.goal_progress_record:
                return False
            
            inner_dict = self.goal_progress_record.get(goal_name, {})
            if not inner_dict:
                return False
            
            try:
                last_progress_value = list(inner_dict.values())[-1][0]
            except (IndexError, TypeError):
                last_progress_value = 0.0
            
            if last_progress_value < self.compliant_threshold:
                return False
        
        self._log_info(f"所有核心目标进度达标（≥{self.compliant_threshold}），可结束问诊", "info")
        return True

    def _log_round_start(self):
        """轮次开始标识"""
        print("-"*60)
        self._log_info("【轮次开始】开始本轮问诊推理", "info")
        print("-"*60)

    def _log_round_end(self):
        """轮次结束标识"""
        self._log_info(f"【轮次结束】本轮完成 | 当前进度记录数: {len(self.goal_progress_record)}", "info")
        print("-"*60)
# 类定义结束