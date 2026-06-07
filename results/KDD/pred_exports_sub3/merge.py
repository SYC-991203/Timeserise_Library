import os
import pandas as pd

# 设置文件夹路径（改为你自己的路径）
folder_path = './'  # 当前目录，也可以写成绝对路径如 'D:/xxx/'

# 定义变量列表
# sub2_vars = ['alpha', 'phi', 'omega', 'P', 'psi']
sub3_vars = ['alpha', 'phi', 'eta', 'P', 'sigma']

# 读取 true.csv
true_df = pd.read_csv(os.path.join(folder_path, 'true.csv'))

# 检查 true_df 是 sub2 还是 sub3 格式
# if set(sub2_vars).issubset(true_df.columns):
#     true_type = 'sub2'
#     true_vars = sub2_vars
if set(sub3_vars).issubset(true_df.columns):
    true_type = 'sub3'
    true_vars = sub3_vars
else:
    raise ValueError("true.csv 中的列名不匹配 sub2 或 sub3 的变量列表")

# 将 true_df 列重命名为 模型_变量_true 的格式，使用 'true' 作为模型名
true_renamed = true_df.rename(columns={var: f'true_{var}_true' for var in true_vars})

# 合并每种子集的模型预测结果
for sub_type, var_list in [('sub3', sub3_vars)]:
    dfs = [true_renamed[[f'true_{var}_true' for var in var_list]]]  # 初始化包含 true 值的 df 列表

    # 遍历所有匹配的模型预测文件
    for file in os.listdir(folder_path):
        if file.endswith(f'{sub_type}.csv') and file != 'true.csv':
            print(f"Processing {file}")
            model_name = file.replace(f'_{sub_type}.csv', '')
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            
            # 仅保留指定变量，并重命名
            renamed_df = df[var_list].rename(columns={var: f'{model_name}_{var}_pred' for var in var_list})
            dfs.append(renamed_df)

    # 合并所有列
    merged_df = pd.concat(dfs, axis=1)

    # 保存为CSV
    output_name = f'merged_{sub_type}.csv'
    merged_df.to_csv(os.path.join(folder_path, output_name), index=False)
    print(f'{output_name} 保存成功！')
