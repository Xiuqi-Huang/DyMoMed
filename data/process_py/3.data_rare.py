#给数据加上是否罕见病的标签

import json
import requests
import time
import os
from tqdm import tqdm  # 进度条库


INPUT_FILE = "pmc700.json"
OUTPUT_FILE = "2rare_strict_pmc700.json"

# --- 常见病黑名单 (根据需要补充) ---
# 这些词如果在病名里出现，直接跳过，节省时间且防误判
BLACKLIST = [
    "gastritis", "hypertension", "diabetes", "influenza", "pneumonia",
    "fracture", "infection", "bronchitis"
]


def check_orphanet_strict(disease_name):
    """
    严格模式查询：
    1. 使用 exact=True
    2. 检查返回名字的长度差异
    """
    if not disease_name: return False, None, "Empty"

    # 1. 黑名单检查
    lower_name = disease_name.lower()
    if any(b in lower_name for b in BLACKLIST):
        return False, None, "Blacklisted"

    url = "https://www.ebi.ac.uk/ols4/api/search"
    params = {
        "q": disease_name,
        "ontology": "ordo",
        "rows": 1,
        "exact": True,  # 必须精确匹配
        "type": "class"  # 只找分类，不找属性
    }

    try:
        # ⚠️ 设置 3秒超时，避免卡死
        resp = requests.get(url, params=params, timeout=3)

        if resp.status_code == 200:
            data = resp.json()
            if data["response"]["numFound"] > 0:
                doc = data["response"]["docs"][0]
                ref_label = doc["label"]

                # --- 智能验证逻辑 ---
                # 如果 API 返回的名字比你查询的名字长太多（超过 2 倍），通常是匹配到了亚型
                # 例如：查 "Anemia" -> 返回 "Fanconi anemia complementation group..."
                if len(ref_label) > len(disease_name) * 2:
                    return False, ref_label, "Mismatch (Length)"

                # 认为是罕见病
                return True, ref_label, "Matched"

    except Exception as e:
        return False, None, f"Error: {str(e)}"

    return False, None, "Not Found"


# def main():
#     if not os.path.exists(INPUT_FILE):
#         print("❌ 找不到输入文件")
#         return

#     with open(INPUT_FILE, "r", encoding="utf-8") as f:
#         records = json.load(f)

#     print(f"🚀 开始处理 {len(records)} 条数据...")
#     print("预计耗时：10-15 分钟（为了保护接口，设置了延迟）\n")

#     valid_rare_count = 0
#     results = []

#     # 使用 tqdm 显示进度条
#     for item in tqdm(records, desc="查询进度"):
#         diag = item.get("Diagnosis", "").strip()

#         # 执行查询
#         is_rare, ref_name, status = check_orphanet_strict(diag)

#         # 更新数据
#         item["Disease_Info"] = {
#             "is_rare": is_rare,
#             "orphanet_name": ref_name if is_rare else None,
#             "status_log": status  # 记录一下为什么是/不是，方便你复查
#         }

#         if is_rare:
#             valid_rare_count += 1

#         results.append(item)

#         # ⚠️ 关键：每次请求后暂停 0.5 秒
#         # 如果不暂停，EBI 服务器会封锁你的 IP
#         time.sleep(0.5)

#     # 保存结果
#     with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#         json.dump(results, f, ensure_ascii=False, indent=4)

#     print("\n" + "=" * 50)
#     print(f"✅ 处理完成！")
#     print(f"共发现罕见病：{valid_rare_count} 例")
#     print(f"结果已保存至：{OUTPUT_FILE}")
#     print("=" * 50)


# if __name__ == "__main__":
#     main()

import json

# 读取数据
with open('2rare_strict_pmc700.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    # 1. 提取原有信息 (pop 会删除原键值对)
    old_info = item.pop("Disease_Info", {})
    dept = item.pop("Department", "未知")
    
    # 2. 重新构建 Disease_Info (剔除 status_log)
    item["Disease_Info"] = {
        "Department": dept,
        "is_rare": old_info.get("is_rare", False),
        "orphanet_name": old_info.get("orphanet_name")
    }

# 保存回文件
with open('rare_pmc700.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("处理完成！格式已优化。")