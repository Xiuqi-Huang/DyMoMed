#用来计算数据中的部门分布的

import json
import os


def detailed_department_analysis(file_path):
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return None

    dept_stats = {}
    total_valid_records = 0

    for item in data:
        # 1. 基础转换（处理双重编码）
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except:
                continue

        if not isinstance(item, dict):
            continue

        # 2. 智能提取科室字段 (尝试多种可能的路径)
        dept = None

        # 路径 A: 根目录下有 Department
        if item.get("Department"):
            dept = item.get("Department")
        # 路径 B: 在 Disease_Info 字典里
        elif isinstance(item.get("Disease_Info"), dict):
            dept = item.get("Disease_Info").get("Department") or item.get("Disease_Info").get("科室")
        # 路径 C: 直接在 Disease_Info 是字符串的情况下（如果数据格式很乱）
        elif isinstance(item.get("Disease_Info"), str) and "科室" in item.get("Disease_Info"):
            # 这种情况比较少见，先占位
            dept = "解析错误(文本格式)"

        # 如果还是没找到
        if not dept:
            dept = "未分类/未知"

        dept_stats[dept] = dept_stats.get(dept, 0) + 1
        total_valid_records += 1

    # --- 统计与输出 (保持不变) ---
    sorted_stats = sorted(dept_stats.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 50)
    print(f"{'科室名称':<20} | {'样本数量':<10} | {'占比 (%)':<10}")
    print("-" * 50)
    for dept, count in sorted_stats:
        percentage = (count / total_valid_records) * 100
        print(f"{dept:<20} | {count:<10} | {percentage:>8.2f}%")
    print("-" * 50)
    print(f"{'总计':<20} | {total_valid_records:<10} | 100.00%")
    print("=" * 50)


if __name__ == "__main__":
    file_path = r'D:\主线\研究生\科研\医疗ai\MedAgent\data\process_document\rare_strict_pmc700.json'
    detailed_department_analysis(file_path)