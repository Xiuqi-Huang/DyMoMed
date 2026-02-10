#用来比较快的从700数据里挑选出100，就是把多的数据随机删掉

import json
import random

# 读取数据
with open('rare_mtmed100.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 筛选出所有 Internal Medicine 的项并随机抽取 40 个
# im_list1 = [d for d in data if d['Disease_Info']['Department'] == 'Internal Medicine']
# to_remove = random.sample(im_list1, 1)


# 过滤数据并重新编号（从 1 开始排到 N）
new_data = [d for d in data]
for i, item in enumerate(new_data, 1):
    item['id'] = i

# 保存新文件
with open('rare_mtmed100.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)

print(f"处理完成，剩余数据量: {len(new_data)}")

import json

# 读取数据
with open('uu_rare_mtmed100.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
#
#
# def should_delete(item):
#     info = item.get("Disease_Info", {})
#     record = item.get("Medical_Record", {})  # 虽然你在例子中写在Clinical_Findings下，但逻辑一致
#     findings = item.get("Clinical_Findings", {})
#
#     # 基础前置条件：Department 是 Surgery 且 is_rare 是 False
#     # 注意：这里匹配的是 Python 布尔值 False
#     if info.get("Department") == "Pediatrics":
#
#         if not findings.get("Physical_Examination") and not findings.get("Investigations"):
#             return True
#
#         # 条件 3: Diagnosis 的词数超过 8 个
#         # 使用 split() 按空格切分单词
#         diagnosis = item.get("Diagnosis", "")
#         if len(diagnosis.split()) > 7:
#             return True
#
#     return False
#
#
# # 执行过滤：保留不满足删除条件的数据
# new_data = [item for item in data if not should_delete(item)]
#
# # 重新排序 ID (从 1 开始)
# for i, item in enumerate(new_data, 1):
#     item['id'] = i
#
# # 保存文件
# with open('uu_rare_mtmed100.json', 'w', encoding='utf-8') as f:
#     json.dump(new_data, f, indent=4, ensure_ascii=False)
#
# print(f"清理完成。原始数量: {len(data)}，剩余数量: {len(new_data)}")

import json
import re

# def is_newborn(demographics_text):
#     """
#     判断文本是否描述的是新生儿（关键词匹配或年龄 < 28天）
#     """
#     text = demographics_text.lower()
#
#     # 1. 关键词直接匹配
#     if any(word in text for word in ['newborn', 'neonate', 'new-born', 'infant']):
#         return True
#
#
# input_file = 'uu_rare_mtmed100.json'
# output_file = 'rare_mtmed100.json'
#
# with open(input_file, 'r', encoding='utf-8') as f:
#     data = json.load(f)
#
# # 过滤数据
# new_data = [
#     item for item in data
#     if not is_newborn(item['Medical_Record'].get('Demographics', ''))
# ]
#
# # 重新对 id 进行 1-N 排序
# for i, item in enumerate(new_data, 1):
#     item['id'] = i
#
# # 保存新文件
# with open(output_file, 'w', encoding='utf-8') as f:
#     json.dump(new_data, f, indent=4, ensure_ascii=False)
#
# print(f"处理完成！")
# print(f"原始数据条数: {len(data)}")
# print(f"剩余数据条数: {len(new_data)}")
# print(f"删除记录数量: {len(data) - len(new_data)}")