import os
import json
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd 
import json
import subprocess
print(os.getcwd())
## 将csv数据转换json 串进行分解
dyg_df = pd.read_csv("./data/DYG/DYG_2_data.csv")

def run_java_command(input_file, output_file):
    command = [
        "/home/syc/source/java/jdk1.8.0_391/bin/java",
        "-jar",
        "java/OneShotSTL/OneShotSTL.jar",
        "--method", "OneShotSTL",
        "--task", "decompose",
        "--shiftWindow", "10",
        "--in", input_file,
        "--out", output_file
    ]
    subprocess.run(command, check=True)  # check=True 会在命令失败时抛出异常


names ={"zt","ht","nn"}
json_list = []
for name in names:
    dyg_dic = {
        "period": 496,
        "trainTestSplit": 1000,
        "ts":  [float(item) for item in dyg_df[name].tolist()]
    }
    file_name = "./data/DYG/json/DYG_{}.json".format(name)
    with open(file_name,"w") as json_file:
        json.dump(dyg_dic,json_file)

### 处理nd nd hx_remake 分解
file_paths =[]
for name in names:
    run_java_command("./data/DYG/json/DYG_{}.json".format(name), "./data/DYG/json/result/DYG_{}_OneShotSTL.json".format(name))
    file_paths.append("./data/DYG/json/result/DYG_{}_OneShotSTL.json".format(name))

### 分解结果可视化
### 分解后的jn,nd,hx——remake的json进行合并

# 文件对应的键名
# 用于存储合并后的数据
merged_data = {}

# 遍历文件路径和键名
for file_path, key in zip(file_paths, names):
    # 读取每个文件的内容
    with open(file_path, 'r') as f:
        data = json.load(f)
    # 将内容添加到merged_data字典中，使用文件名作为键
    merged_data[key] = data

# 将合并后的字典写入新的JSON文件中


cols = names
total_df = pd.DataFrame()
for col in cols:
    trend = merged_data[col]["trend"]
    seasonal = merged_data[col]["seasonal"]
    residual = merged_data[col]["residual"]
    df = pd.DataFrame({"{}_trend".format(col):trend,
                      "{}_seasonal".format(col):seasonal,
                      "{}_residual".format(col):residual
                      })
    total_df = pd.concat([total_df,df],axis=1)
dates = pd.date_range(start='2021-01-01 00:00', periods=len(df), freq='H')

# 首先将日期时间格式化为"YYYY-MM-DD HH:MM"
dates_formatted = dates.strftime('%Y-%m-%d %H:%M')

# 然后将格式化后的日期时间字符串转换为所需的格式
# 这里需要先将字符串转换为datetime对象，然后再进行格式化
dates_final = [pd.to_datetime(date).strftime('%Y/%-m/%-d %H:%M') for date in dates_formatted]


# 将格式化后的时间序列添加为新列
total_df['date'] = dates_final

total_df.to_csv("./data/DYG/DYG_zhn_Oneshot.csv",index=False)
print(total_df.head(10))
