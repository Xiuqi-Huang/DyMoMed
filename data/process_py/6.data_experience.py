#用来挑出700条数据中不是100条数据的那些，并且选了诊断长度<=6的作为可用的经验库，之后再加上情绪

import json


def get_difference_and_filter(file_a, file_b, output_file):
    """
    找出在 A 中但不在 B 中的数据，并过滤掉 Diagnosis 词数 > 10 的项
    """
    try:
        # 1. 加载数据
        with open(file_a, 'r', encoding='utf-8') as f:
            data_a = json.load(f)
        with open(file_b, 'r', encoding='utf-8') as f:
            data_b = json.load(f)

        # 2. 提取 B 中所有数据的特征作为“指纹”存入集合，用于高效比对
        # 即使 ID 变了，只要这三个文本没变，就认为是同一条数据
        b_fingerprints = set()
        for item in data_b:
            mr = item.get("Medical_Record", {})
            fingerprint = (
                mr.get("Chief_Complaint"),
                mr.get("Background"),
                mr.get("History_of_Present_Illness")
            )
            b_fingerprints.add(fingerprint)

        # 3. 筛选 A 中不在 B 且满足词数要求的数据
        result = []
        for item in data_a:
            mr = item.get("Medical_Record", {})
            current_fingerprint = (
                mr.get("Chief_Complaint"),
                mr.get("Background"),
                mr.get("History_of_Present_Illness")
            )

            # 如果不在 B 文件中
            if current_fingerprint not in b_fingerprints:
                # 检查 Diagnosis 词数
                diagnosis = item.get("Diagnosis", "")
                word_count = len(diagnosis.split())

                # 只有词数 <= 10 才保留
                if word_count <= 6:
                    result.append(item)

        # 4. 保存新文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        print(f"成功生成: {output_file} | 提取数据量: {len(result)}")

    except FileNotFoundError as e:
        print(f"错误: 找不到文件 {e.filename}，请检查路径。")
    except Exception as e:
        print(f"处理 {file_a} 时发生错误: {e}")


# --- 运行 3 次的逻辑 ---

# 请在这里修改您的文件名对
tasks = [
    # (原始文件A, 抽样文件B, 输出结果文件)
    ("rare_clinic214.json", "rare_clinic100.json", "exp_clinic.json"),
    ("rare_mtmed700.json", "rare_mtmed100.json", "exp_mtmed.json"),
    ("rare_pmc700.json", "rare_pmc100.json", "exp_pmc.json")
]

for a, b, out in tasks:
    get_difference_and_filter(a, b, out)