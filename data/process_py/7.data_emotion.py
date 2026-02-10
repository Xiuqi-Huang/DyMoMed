#用于向数据中添加情绪和说话方式，顺便确保每一项里的顺序是相同的

import json, random

#文件路径
INPUT = r"exp_pmc.json"
OUTPUT = "exp_pmc.json"

EMOTIONS = [
    ("Calm", "Responds objectively, focuses on facts, and keeps a steady tone."), # 冷静：客观回应，专注于陈述事实，并保持平稳的语调。
    ("Anxious", "Seeks reassurance, asks repetitive questions, and fixates on worst-case scenarios."), # 焦虑：不断寻求安慰，重复提问，并且总是盯着最坏的情况（杞人忧天）。
    ("Frustrated", "Complains about symptoms, challenges the doctor's questions, and sounds irritable."), # 懊恼/受挫：抱怨身体症状，质疑医生提问的必要性，听起来很烦躁易怒。
    ("Denial", "Downplays pain, insists symptoms are minor, and dismisses severity."), # 否认/回避：轻描淡写疼痛，坚持认为症状只是小毛病，无视病情的严重性。
    ("Optimistic", "Focuses on recovery, expresses hope, and uses positive language."),# 乐观：关注康复前景，表达希望，并使用积极正面的语言。
    ("Skeptical", "Doubts the diagnosis, questions the doctor's logic, and asks for proof.")# 怀疑：怀疑医生的诊断，质问医生的逻辑，并要求提供证据。
]

STYLES = [
    ("Concise", "Uses short sentences, answers strictly 'yes' or 'no', and avoids elaboration."), # 简洁：使用短句，严格只回答“是”或“否”，避免展开解释。
    ("Verbose", "Over-explains, includes irrelevant backstory, and rambles off-topic."), # 话痨/冗长：过度解释，夹带无关的背景故事，说话跑题且絮叨。
    ("Hesitant", "Uses fillers like 'um' or 'maybe', pauses often, and avoids direct commitment."), # 犹豫：使用“嗯”、“大概”等填充词，经常停顿，避免给出确切的承诺或回答。
    ("Forgetful", "Mixes up dates, frequently says 'I don't recall', and gives vague details."),# 健忘：搞混日期，经常说“我不记得了”，提供的细节非常模糊。
    ("Confused", "Misunderstands questions, asks for repetition, and gives unrelated answers.")# 困惑：误解医生的问题，要求医生重复，给出的回答答非所问。
]

# random.seed(2026) #针对clinic
# random.seed(611) #针对mtmed
random.seed(29) #针对pmc

with open(INPUT, "r", encoding="utf-8") as fin:
    data_list = json.load(fin)

for item in data_list:
    mr = item.get("Medical_Record", {})
    cf = item.get("Clinical_Findings", {})
    e, ed = random.choice(EMOTIONS)
    s, sd = random.choice(STYLES)

    # 构造Personality
    personality_str = f"Emotion: {e} ({ed}), Speaking Style: {s} ({sd})"

    item["Medical_Record"] = {
        "Demographics": mr.get("Demographics"),
        "Personality": personality_str,
        "Chief_Complaint": mr.get("Chief_Complaint"),
        "History_of_Present_Illness": mr.get("History_of_Present_Illness"),
        "Background": mr.get("Background")}

    item["Clinical_Findings"] = {
        "Physical_Examination": cf.get("Physical_Examination"),
        "Investigations": cf.get("Investigations")}

    ordered_item = {
        "id": item.get("id"),
        "Medical_Record": item["Medical_Record"],
        "Clinical_Findings": item["Clinical_Findings"],
        "Diagnosis": item.get("Diagnosis"),
        "Disease_Info": item.get("Disease_Info")}
    
    # 将重排后的对象写回列表（原地修改）
    item.clear()
    item.update(ordered_item)

with open(OUTPUT, "w", encoding="utf-8") as fout:
    json.dump(data_list, fout, indent=4, ensure_ascii=False)

print("OK")
