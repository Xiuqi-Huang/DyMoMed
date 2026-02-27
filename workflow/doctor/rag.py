import os
import json
from typing import List, Dict
from datetime import datetime
import logging
from pathlib import Path
import time
import requests
import xml.etree.ElementTree as ET
from langchain_core.prompts import PromptTemplate
import httpx
from openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
import re  

# 替换导入语句
from langchain_chroma import Chroma
from langchain_core.documents import Document
import wikipedia
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import numpy as np
import threading
import os



import chromadb
from chromadb.config import Settings
rag_lock = threading.Lock()

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import re  
import os
import time
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def query_model_ours(client, prompt, system_prompt, config_json: dict = None, tries=2, timeout=20.0, max_prompt_len=2**14, clip_prompt=False):
 
    # 加载JSON配置，覆盖默认参数
    config = config_json or {}
    model_str = config.get("model_str", "gpt4")
    tries = config.get("tries", tries)
    timeout = config.get("timeout", timeout)
    max_prompt_len = config.get("max_prompt_len", max_prompt_len)
    clip_prompt = config.get("clip_prompt", clip_prompt)
    temperature = config.get("temperature", 0.01)
    response_format = config.get("response_format", {"type": "text"})  # 新增支持response_format

    valid_models = ["gpt4"]
    # 保存并清空代理
    old_http_proxy = os.environ.get("http_proxy", "")
    old_https_proxy = os.environ.get("https_proxy", "")
    old_no_proxy = os.environ.get('NO_PROXY', '')
    
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""
    os.environ['NO_PROXY'] = ''

    # 校验模型合法性
    if model_str not in valid_models and "_HF" not in model_str:
        os.environ["http_proxy"] = old_http_proxy
        os.environ["https_proxy"] = old_https_proxy
        os.environ['NO_PROXY'] = old_no_proxy
        logger.error(f"无效模型名称: {model_str}")
        return None, 0

    try:
        for _ in range(tries):
            # 截断超长提示词
            if clip_prompt and len(prompt) > max_prompt_len:
                prompt = prompt[:max_prompt_len]
                logger.warning(f"提示词超长，已截断至{max_prompt_len}字符")
            
            try:
                total_tokens = 0
                if model_str == "gpt4":
                    # 调用OpenAI API（新增response_format参数）
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                        temperature=temperature,
                        response_format=response_format  # 传递JSON格式要求
                    )
                    # 解析返回结果
                    content = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
                    answer = re.sub(r"[ \t]+", " ", content)
                    # 统计token数
                    if response.usage:
                        total_tokens = response.usage.total_tokens
                    return answer, total_tokens
                elif "HF_" in model_str:
                    logger.error("HF models 暂未实现")
                    return None, 0
            except Exception as e:
                logger.error(f"[API调用失败] 重试中 - {type(e).__name__}: {e}")
                time.sleep(timeout)
                continue
        logger.error("达到最大重试次数，调用超时")
        return None, 0
    finally:
        # 恢复代理配置
        os.environ["http_proxy"] = old_http_proxy
        os.environ["https_proxy"] = old_https_proxy
        os.environ['NO_PROXY'] = old_no_proxy
        
        
class CustomSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model = SentenceTransformer(model_path, device=device)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()
    
    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

class PubMedAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.requests_made = 0
        self.last_request_time = time.time()
        # 新增：最大等待时间限制（避免无限sleep）
        self.max_sleep_time = 5

    
    def throttle(self):
        max_requests = 10 if self.api_key else 3
        current_time = time.time()
        
        # 重置计数：超过1秒则重置
        if current_time - self.last_request_time > 1.0:
            self.requests_made = 0
            self.last_request_time = current_time

        # 检查请求数是否超限
        if self.requests_made >= max_requests:
            # 计算需要sleep的时间（最多self.max_sleep_time）
            sleep_time = min(1.0 - (current_time - self.last_request_time), self.max_sleep_time)
            if sleep_time > 0:
                logger.info(f"PubMed请求频率超限，休眠{sleep_time:.2f}秒")
                time.sleep(sleep_time)
            # 重置计数
            self.requests_made = 0
            self.last_request_time = time.time()

        # 计数+1并更新时间
        self.requests_made += 1
        self.last_request_time = current_time

    # 搜索PubMed文献
    def search_pubmed(self, query, max_results = 1):
        self.throttle()
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "xml",
            "retmax": max_results,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            # 添加超时（10秒），避免网络阻塞
            response = requests.get(base_url, params=params, timeout=10) 
            response.raise_for_status()
            root = ET.fromstring(response.text)
            return [id_elem.text for id_elem in root.findall(".//Id")]
        except requests.exceptions.Timeout:
            logger.error(f"PubMed搜索超时（query: {query}）")
            return []
        except Exception as e:
            logger.error(f"PubMed搜索失败（query: {query}）: {e}")
            return []

    def fetch_articles(self, pmids):
        if not pmids:
            return []
        self.throttle()
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "text",
            "rettype": "abstract",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            # 添加超时（10秒）
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            return response.text.split("\n\n")
        except requests.exceptions.Timeout:
            logger.error(f"PubMed获取摘要超时（pmids: {pmids}）")
            return []
        except Exception as e:
            logger.error(f"PubMed获取摘要失败（pmids: {pmids}）: {e}")
            return []

# 核心RAG类
class RAG:
    def __init__(self, persist_directory: str = "", goal_json_path: str = "goals.json"):
        # 初始化目标集（从JSON读取，无则创建默认）
        self.goal_json_path = Path(goal_json_path)
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # 加载/初始化目标集
        self.goals = self._load_goals_from_json()
        
        # 初始化嵌入模型
        print("开始文本嵌入模型初始化.")
        try:
            # 优先使用自定义嵌入模型（兼容本地模型）
            self.embeddings = CustomSentenceTransformerEmbeddings(
                model_path=r"D:\\ZM\\01-KDD\\MedAgent-main\\workflow\\all-mpnet-base-v2",
                device="cpu"
            )
        except:
            # 降级使用HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=r"D:\\ZM\\01-KDD\\MedAgent-main\\workflow\\all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'},
            )
        print("已完成文本嵌入模型初始化.")
        
        # 初始化向量数据库
        self.vectorstore = self._initialize_vectorstore()
        
        # 元数据管理
        self.metadata_file = self.persist_directory / "metadata.json"
        self.metadata = self._load_metadata()

        # 初始化外部知识组件
        self.pubmed_api = PubMedAPI(api_key=os.environ.get('PUBMED_API_KEY'))
        self.max_wiki_search = 0
        self.max_pubmed_search = 1

        self.external_k_llm = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url="",
            http_client=httpx.Client(verify=True, timeout=httpx.Timeout(300.0, connect=60.0), follow_redirects=True)
        )

    # 新增方法：从向量库直接提取所有source_file
    def get_all_source_files_from_vdb(self):
        """
        直接从Chroma向量库读取所有文档的source_file，返回去重后的文件列表
        """
        with rag_lock:
            try:
                # 1. 获取向量库中所有文档（含元数据）
                # get()方法会返回所有文档的元数据和内容（适配langchain-chroma）
                all_docs = self.vectorstore.get()
                
                # 2. 提取所有source_file字段
                source_files = []
                # all_docs['metadatas'] 是所有文档的元数据列表
                for metadata in all_docs.get('metadatas', []):
                    source_file = metadata.get('source_file')
                    if source_file:  # 过滤空值
                        source_files.append(source_file)
                
                # 3. 去重并返回
                unique_source_files = list(set(source_files))
                print(f"向量库中实际存储的文件列表（去重后）：{unique_source_files}")
                print(f"总计涉及 {len(unique_source_files)} 个文件")
                
                # 可选：统计每个文件的chunk数量
                file_chunk_count = {}
                for sf in source_files:
                    file_chunk_count[sf] = file_chunk_count.get(sf, 0) + 1
                print("每个文件的chunk数量：", file_chunk_count)
                
                return unique_source_files
            
            except Exception as e:
                print(f"读取向量库source_file失败: {e}")
                return []
    
    # 从JSON加载目标集
    def _load_goals_from_json(self):
        
        goals = {
    "nodes": [
        {
            "name": "Diagnostic Accuracy",
            "description": "Extract key information from the given data to complete differential diagnosis and provide rigorous and accurate diagnostic conclusions.",
            "score": 3
        },
        {
            "name": "Safety",
            "description": "Conduct a comprehensive assessment based on symptoms, medical history and examination results. Strictly prohibit the omission of life-threatening etiologies and critical high-risk clinical clues, and formulate safe and implementable treatment plans.",
            "score": 3
        },
        {
            "name": "Treatment Comprehensiveness",
            "description": "Treatment plans shall emphasize comprehensiveness. In addition to suggestions for symptomatic treatment of benign diseases, consideration shall also be given to possible urgent examinations, specialist referral recommendations, and indications for emergency department visits.",
            "score": 3
        },
        {
            "name": "Patient Emotional Care",
            "description": "Pay attention to the patient's emotional state during the medical consultation, provide timely comfort to ensure smooth communication.",
            "score": 3
        }
    ],
    "edges": [
        ("Diagnostic Accuracy", "", "comp"),
        ("Treatment Comprehensiveness", "", "comp"),
        ("Safety", "", "comp"),
        ("Patient Emotional Care", "", "comp")
    ]
}
        
        # 1. 无JSON文件：创建空文件（而非默认内容），返回空目标集
        if not self.goal_json_path.exists():
            self.save_goals_to_json(goals)  # 初始化空文件
            print(f"目标集文件不存在，已创建空文件: {self.goal_json_path}")
            return goals
        
        # 2. 文件存在但为空（大小为0）：直接返回空目标集，不解析、不触发异常
        if self.goal_json_path.stat().st_size == 0:
            print(f"目标集JSON文件存在但内容为空，返回空目标集")
            return goals
        
        # 3. 文件非空：尝试解析，解析失败时返回空目标集（而非默认值）
        try:
            with open(self.goal_json_path, 'r', encoding='utf-8') as f:
                loaded_goals = json.load(f)
                # 校验结构合法性（确保是dict且有nodes/edges，避免格式乱）
                if not isinstance(loaded_goals, dict) or "nodes" not in loaded_goals or "edges" not in loaded_goals:
                    print("目标集JSON结构不合法，返回空目标集")
                    return goals
                return loaded_goals
        except json.JSONDecodeError as e:
            print(f"目标集JSON格式错误: {e}，返回空目标集")
            return goals
        except Exception as e:
            print(f"读取目标集JSON异常: {e}，返回空目标集")
            return goals

    # 保存目标集到JSON（空结构也能正常保存）
    def save_goals_to_json(self, goals_dict: Dict):
        try:
            # 确保父目录存在（避免路径不存在导致保存失败）
            self.goal_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.goal_json_path, 'w', encoding='utf-8') as f:
                json.dump(goals_dict, f, indent=4, ensure_ascii=False)
            print(f"目标集已保存到: {self.goal_json_path}（当前内容：{goals_dict}）")
        except Exception as e:
            print(f"保存目标集JSON失败: {e}")

    # 初始化向量数据库
    
    def _initialize_vectorstore(self) -> Chroma:
        
        persist_dir = str(self.persist_directory)
        if (self.persist_directory / "chroma.sqlite3").exists():
            print("Loading existing vector database...")
            return Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings,
                collection_name="experience"
            )
        else:
            print("Creating new vector database...")
            return Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings,
                collection_name="experience"
            )
        # 加载元数据
    def _load_metadata(self) -> Dict:
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "documents": {},
            "last_updated": None,   
            "total_rounds": 0, 
            "sources": [],
            "total_chunks": 0
        }

    # 更新元数据
    def _update_metadata(self):
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    # 格式化回合文本
    def _format_round_text(self, round_data: Dict, bench_name: str, init_goals: List) -> str:
        """
        格式化单轮经验文本（优化点：加入核心检索维度、控制长度、清理冗余空白）
        :param round_data: 单轮经验数据
        :param bench_name: 问诊场景（如"内科腹痛问诊"）
        :param init_goals: 初始目标列表
        :return: 格式化后的纯文本（适合向量嵌入）
        """
        # 配置：单轮文本最大长度（适配text-embedding-3-small的token限制）
        MAX_TEXT_LENGTH = 2000
        # 提取并清理字段（去除首尾空白+长度截断）
        previous_summary = round_data.get('previous_summary', '').strip()[:MAX_TEXT_LENGTH]
        current_dialogue = round_data.get('current_dialogue', '').strip()[:MAX_TEXT_LENGTH]
        current_reasoning = round_data.get('current_reasoning', '').strip()[:MAX_TEXT_LENGTH]
        current_evaluation = round_data.get('current_evaluation', '').strip()[:MAX_TEXT_LENGTH]
        round_id = round_data.get('round_id', '')
        # 初始目标转为字符串（长度限制+去空）
        init_goals_str = ", ".join([g.get('name', '').strip() for g in init_goals if g.get('name')])[:500]

        # 构造格式化文本（无冗余缩进/空行，融入核心检索维度）
        formatted_text = (
            f"Consultation scenario: {bench_name}\n"
            f"Initial goals: {init_goals_str}\n"
            f"Summary of previous conversation history: {previous_summary}\n"
            f"Conversation in the current round {round_id}: {current_dialogue}\n"
            f"Doctor's reasoning outcome in the current round {round_id}: {current_reasoning}\n"
            f"Evaluation in the current round {round_id}: {current_evaluation}"
        ).strip()

        # 最终长度兜底（避免超长）
        return formatted_text[:MAX_TEXT_LENGTH]

    # 优化后的单文档入库函数
    def ingest_single_document(self, file_path: str):
        """
        导入单篇经验JSON文档到向量库（优化：存储目标名称而非完整字典，适配检索逻辑）
        :param file_path: 经验JSON文件路径
        :return: None
        """
        print(f"Ingesting single document: {file_path}")
        source_name = Path(file_path).name

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 1. 提取初始目标的名称列表（从字典列表中抽name）
            init_goals = data.get('init_goals', [])
            init_goal_names = [g.get('name', '') for g in init_goals if g.get('name')]  

            rounds = data.get("rounds", [])
            bench_name = data.get('bench_name', '')
            
            case_metadata = {}
            for k, v in data.items():
                if k == 'rounds':
                    continue
                
                if isinstance(v, (list, dict)):
                    case_metadata[k] = json.dumps(v, ensure_ascii=False)
                else:
                    case_metadata[k] = v

            chunks = []
            for i, round_data in enumerate(rounds):
                # 2. 格式化轮次文本
                round_text = self._format_round_text(round_data, bench_name, init_goals)

                # 3. 提取本轮new_goals
                round_new_goal_names = round_data.get('new_goals', [])  # 现在是["目标名称1", ...]
                # 4. 本轮所有目标名称（初始+本轮新增）
                round_goal_names = round_new_goal_names
                round_goal_names = [name for name in round_goal_names if name]  # 过滤空名称
                round_goals_str = json.dumps(round_goal_names, ensure_ascii=False)  # 存储名称列表

                # 5. 处理goals_performance
                goals_performance = round_data.get('goals_performance', {})
                goals_performance_str = json.dumps(goals_performance, ensure_ascii=False)

                # 6. 构造Document（元数据仅存名称，适配检索）
                file_suffix = Path(file_path).suffix.lower()
                doc = Document(
                    page_content=round_text,
                    metadata={
                        'chunk_id': self.metadata['total_chunks'] + i,
                        'ingestion_date': datetime.now().isoformat(),
                        'source_file': source_name,
                        'goals_round': round_goals_str,          # 仅存储目标名称列表JSON
                        'goals_performance': goals_performance_str,
                        'document_type': file_suffix.lstrip('.'),
                        'round_index': round_data.get('round_id', i),
                        'total_rounds': len(rounds),
                    }
                )
                doc.metadata.update(case_metadata)                
                chunks.append(doc)

        except json.JSONDecodeError as e:
            print(f"解析JSON文件{file_path}失败: {e}")
            return None
        except Exception as e:
            print(f"导入文件{file_path}出错: {e}")
            return None

    
        #print(f"chunks是否为空{chunks}")
        if chunks:
            self.vectorstore.add_documents(chunks)
            # self.vectorstore.persist()

            if source_name not in self.metadata['sources']:
                self.metadata['sources'].append(source_name)

            self.metadata['total_chunks'] += len(chunks)
            self.metadata['last_updated'] = datetime.now().isoformat()
            self.metadata['documents'][source_name] = {
                'chunks': len(chunks),
                'ingestion_date': datetime.now().isoformat(),
                'type': Path(file_path).suffix.lstrip('.')
            }

            self._update_metadata()
            print(f"成功从{source_name}导入{len(chunks)}个文本块")

        return None
            # 推理时的经验检索（核心复用函数）
    def retrieve_for_goal_reasoning(self, query_dict: Dict, k_per_goal: int = 4, top_k: int = 2) -> Dict[str, List[Document]]:
        with rag_lock:  # 加锁
            print(f"开始目标推理检索: 目标数量={len(query_dict.get('current_goals', []))}")

            current_goals = query_dict.get('current_goals', [])  # 现在是目标名称列表
            previous_dialogue = query_dict.get('previous_dialogue', '')
            query_text = f"Previous dialogue: {previous_dialogue}"
            results_by_goal = {}

            for goal_name in current_goals:
                if not goal_name:
                    print("空目标名称，跳过检索")
                    results_by_goal[goal_name] = []
                    continue
                
                try:
                    # 优化：先检索所有候选，再内存中精准筛选（避免向量库filter语法差异）
                    candidates_with_score = self.vectorstore.similarity_search_with_score(
                        query_text,
                        k=k_per_goal * 2,  # 先取更多候选
                        
                    )

                    if not candidates_with_score:
                        print(f"目标 '{goal_name}' 未找到相关记录")
                        results_by_goal[goal_name] = []
                        continue
                    
                    
                    filtered_candidates = []
                    for doc, score in candidates_with_score:
                        try:
                            # 反序列化goals_round（目标名称列表）
                            goals_round = json.loads(doc.metadata.get('goals_round', '[]'))
                            #print(f"这里：{goals_round}\n\n")
                            
                            goals_round_name = [goal['name'] for goal in goals_round]
                            #print(f"这里2：{goals_round_name}")
                            # print(f"goal_name：{goal_name}\ngoals_round_name：{goals_round_name}\n{goal_name in goals_round_name}")
                            if goal_name in goals_round_name:
                                filtered_candidates.append((doc, score))
                        except json.JSONDecodeError:
                            continue  # 序列化异常则跳过
                        
                    if not filtered_candidates:
                        print(f"目标 '{goal_name}' 无匹配的经验记录")
                        results_by_goal[goal_name] = []
                        continue
                    
                    # 结合表现值排序
                    scored_docs = []
                    for doc, score in filtered_candidates:
                        performance_dict = doc.metadata.get('goals_performance', '{}')
                        try:
                            performance_dict = json.loads(performance_dict)
                        except json.JSONDecodeError:
                            performance_dict = {}

                        goal_performance = performance_dict.get(goal_name, 0)

                        doc_copy = Document(
                            page_content=doc.page_content,
                            metadata=doc.metadata.copy()
                        )
                        doc_copy.metadata['similarity_score'] = float(score)
                        doc_copy.metadata['goal_performance'] = goal_performance
                        doc_copy.metadata['goal'] = goal_name
                        doc_copy.metadata['combined_score'] = (float(score) + goal_performance) / 2

                        scored_docs.append(doc_copy)

                    # 按综合分数排序取Top-K
                    scored_docs.sort(key=lambda x: x.metadata['combined_score'], reverse=True)
                    top_docs = scored_docs[:top_k]
                    results_by_goal[goal_name] = top_docs

                    # print(f"\nresults_by_goal：{results_by_goal}\n")

                except Exception as e:
                    print(f"检索目标 '{goal_name}' 时出错: {e}")
                    results_by_goal[goal_name] = []

            return results_by_goal
            # 回复时的经验检索（核心复用函数）
    def retrieve_for_response(self, previous_dialogue: str, k: int = 2) -> List[Document]:
        with rag_lock: 
            print(f"开始回复时检索")
    
            try:
                query_text = f"Previous dialogue: {previous_dialogue}"
                expanded_k = k * 3
                candidates_with_score = self.vectorstore.similarity_search_with_score(
                    query_text,
                    k=expanded_k,
                )
    
                if not candidates_with_score:
                    print("未检索到相关文档")
                    return []
    
                # 综合相似度+表现值+目标数排序
                scored_docs = []
                goal_counts = [len(doc.metadata.get('goals_round', [])) for doc, _ in candidates_with_score]
                max_goal_count = max(goal_counts) if goal_counts else 1
    
                for i, (doc, similarity_score) in enumerate(candidates_with_score):
                    similarity = float(similarity_score)
                    performance_dict = doc.metadata.get('goals_performance', {})
                    try:
                        performance_dict = json.loads(performance_dict)  # 新增：字符串转字典
                    except json.JSONDecodeError:
                        performance_dict = {}
                    avg_performance = sum(performance_dict.values()) / len(performance_dict) if performance_dict else 0
                    
                    goal_count = goal_counts[i] if i < len(goal_counts) else 0
    
                    # 计算综合分数
                    goal_count_factor = goal_count / max_goal_count
                    mid_performance = avg_performance * goal_count_factor
                    composite_score = 0.5 * similarity + 0.5 * mid_performance
    
                    enhanced_doc = Document(
                        page_content=doc.page_content,
                        metadata={
                            **doc.metadata,
                            'similarity_score': similarity,
                            'goal_count': goal_count,
                            'avg_performance': avg_performance,
                            'mid_performance': mid_performance,
                            'composite_score': composite_score,
                        }
                    )
                    scored_docs.append(enhanced_doc)
    
                # 排序取Top-K
                scored_docs.sort(key=lambda x: x.metadata['composite_score'], reverse=True)
                final_results = scored_docs[:k]
                print(f"回复检索完成: 返回{len(final_results)}个文档")
    
                return final_results
    
            except Exception as e:
                print(f"回复检索过程中出错: {e}")
                return []
    
    
    def gen_search_list(self, query):
        
        print("调用gen_search_list.\n")
        try:
            # 从query中拆分system_prompt和user_prompt（适配query_model_ours参数格式）
            system_prompt = ""
            user_prompt = ""
            for msg in query:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                elif msg["role"] == "user":
                    user_prompt = msg["content"]

            # 调用query_model_ours
            config_json = {
                "model_str": "gpt4",
                "tries": 2,
                "timeout": 20.0,
                "temperature": 0.1,  
                "clip_prompt": False,
                "response_format": {"type": "json_object"}  
            }
            # 调用LLM并获取结果
            llm_response, tokens_cost = query_model_ours(
                client=self.external_k_llm,
                prompt=user_prompt,
                system_prompt=system_prompt,
                config_json=config_json
            )

            # 解析返回结果
            search_list = json.loads(llm_response.strip())
            return {"search_list": search_list.get("search_phrases", [])}, tokens_cost
        except Exception as e:
            print(f"生成搜索词异常: {e}")
            return {"search_list": []}, 0

    def get_pubmed_results(self, search_list):
        print("---External Search PubMed---")
        items = search_list.get("search_list", [])
        individual_results = []
        if not items:
            return {"searchs": "No search results"}

        for item in items:
            try:
                pmids = self.pubmed_api.search_pubmed(item, max_results=self.max_pubmed_search)
                articles = self.pubmed_api.fetch_articles(pmids)
                filter_keywords = ["author", "doi", "pmid", "conflict of interest", "copyright"]
                filtered_articles = []
                for article in articles:
                    main_content = []
                    for line in article.split("\n"):
                        if line.strip() and not any(keyword.lower() in line.lower() for keyword in filter_keywords):
                            main_content.append(line.strip())
                    if main_content:
                        filtered_articles.append(" ".join(main_content))
                combined_content = "\n\n".join(filtered_articles)
                individual_results.append({"search_item": item, "content": combined_content})
            except Exception as e:
                print(f"PubMed检索异常 {item}: {e}")
                individual_results.append({"search_item": item, "content": "Error in retrieving content"})

        return {"searchs": individual_results}