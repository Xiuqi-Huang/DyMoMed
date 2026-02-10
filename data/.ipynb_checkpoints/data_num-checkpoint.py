#这个代码是从原始数据集中随机抽取数据并且重新保存为新文件

import json
import random

def extract_cases(input_path, output_path, k=700):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. 随机打乱并抽取 k 条数据
    sampled_data = random.sample(data, min(k, len(data)))

    # 2. 仅保留指定的字段
    keys_to_keep = ["title", "patient", "age", "gender"]
    
    # 3. 修改保存逻辑：以 JSONL 格式写入
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sampled_data:
            # 提取指定字段
            # filtered_item = {key: item.get(key) for key in keys_to_keep} #仅PMC要用
            # 将字典转为字符串并写入一行，ensure_ascii=False 保证中文/符号正常显示
            # f.write(json.dumps(filtered_item, ensure_ascii=False) + '\n') #PMC用这个
            f.write(json.dumps(item, ensure_ascii=False) + '\n') #MTMed用这个

    print(f"成功随机抽取 {len(sampled_data)} 条数据并保存为 JSONL 格式至 {output_path}")

if __name__ == "__main__":
    # 执行抽取
    # extract_cases("data/PMC-Patients-V2.json", "data/PMC700.jsonl",k=700)
    extract_cases("MTMedDialog_test.json", "MTMed700.jsonl",k=700)