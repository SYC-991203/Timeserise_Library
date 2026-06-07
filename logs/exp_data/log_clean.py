import os
import re
import pandas as pd
import glob

def parse_logs_strict_group0(log_folder_path, output_folder_path):
    # 1. 检查并创建输出目录
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # 2. 获取所有 .log 文件
    log_files = glob.glob(os.path.join(log_folder_path, "*.log"))
    
    if not log_files:
        print(f"❌ 在 {log_folder_path} 未找到 .log 文件")
        return

    # 正则表达式：在一行中匹配所有出现的 Group, Prob, Alpha
    # 解决日志粘连问题 (例如 ...Alpha:0.5[MOE_TRACE]...)
    pattern = re.compile(r"\[MOE_TRACE\]\s+Group:(\d+)\s+Prob:([\d\.]+)\s+Alpha:([\d\.]+)")

    for log_file in log_files:
        file_name = os.path.basename(log_file)
        print(f"处理文件: {file_name} ...")
        
        # 初始化数据存储
        columns_data = {
            0: [], 1: [], 2: [], 3: [], 4: [],
            'Alpha': [] 
        }

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue 

                    # 使用 findall 提取该行所有匹配项
                    matches = pattern.findall(line)

                    for group_id_str, prob_str, alpha_str in matches:
                        group_id = int(group_id_str)
                        prob_val = float(prob_str)
                        alpha_val = float(alpha_str)

                        # 1. 始终记录各组的 Probability
                        if group_id in columns_data:
                            columns_data[group_id].append(prob_val)
                        
                        # 2. 【核心逻辑】Alpha 只能 Group 0 对齐
                        # 只有当当前数据是 Group 0 时，才记录 Alpha
                        # 其他组 (1,2,3,4) 的 Alpha 值会被直接丢弃
                        if group_id == 0:
                            columns_data['Alpha'].append(alpha_val)

        except Exception as e:
            print(f"⚠️ 读取错误: {e}")
            continue

        # === 数据对齐处理 ===
        # 虽然我们逻辑上是对齐 Group 0，但如果日志有缺失（比如 Group 4 打印少了），
        # 我们仍需找出最长的一列，用空值补齐，防止生成 CSV 报错。
        
        max_len = max(len(v) for v in columns_data.values())
        
        final_dict = {}
        
        # 填充 Group 列
        for g_id in [0, 1, 2, 3, 4]:
            col = columns_data[g_id]
            if len(col) < max_len:
                col.extend([None] * (max_len - len(col)))
            final_dict[f'Group_{g_id}'] = col
            
        # 填充 Alpha 列
        # 如果 Group 0 少于其他组（异常情况），Alpha 也会短，同样补齐
        alpha_col = columns_data['Alpha']
        if len(alpha_col) < max_len:
            alpha_col.extend([None] * (max_len - len(alpha_col)))
        final_dict['Alpha'] = alpha_col

        # 生成并保存
        df = pd.DataFrame(final_dict)
        csv_name = file_name.replace('.log', '.csv')
        save_path = os.path.join(output_folder_path, csv_name)
        df.to_csv(save_path, index=False)
        print(f"✅ 保存成功: {save_path}")

# --- 运行配置 ---
# 请修改为你的实际路径
input_dir = '/home/home_new/syc/code/Timeserise_Library/logs/exp_data'    
output_dir = '/home/home_new/syc/code/Timeserise_Library/logs/exp_data/csv_output' 

if __name__ == '__main__':
    parse_logs_strict_group0(input_dir, output_dir)