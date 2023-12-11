import numpy as np
import pandas as pd
import os
current_directory = os.getcwd()
print(current_directory)
subdir = "results"
subdir_path  = os.path.join(current_directory, subdir)
file_paths = [os.path.join(subdir_path, filename) for filename in os.listdir(subdir_path) ## 可以锁定到long_term
              if os.path.isdir(os.path.join(subdir_path, filename))] 

dic = ["/metrics","/pred","/true"]
# 打印所有子文件的路径
for path in file_paths:
    for item in dic:
        data_path = path+item+".npy"
        if ".py" not in data_path:
            data = np.load(data_path)
            if "metric" in data_path:
                column_names = ["mae", "mse", "rmse", "mape", "mspe"]
                # 创建 DataFrame
                df = pd.DataFrame({column_names[i]: [data[i]] for i in range(len(data))})
                df.to_csv(path+"/metric.csv",sep="\t")
                #print(data)
                #df = pd.DataFrame()
            if "pred" in data_path or "true" in data_path: ## 目前这个里面有silde window的问题，细想一下好像不用解决，能对齐就行
                num_features = data.shape[-1]
                column_names = [f'Feature_{i+1}' for i in range(num_features)]
                # 将三维数组转换为 DataFrame
                df = pd.DataFrame(data.reshape(-1, num_features), columns=column_names)
                if "pred" in data_path:
                    df.to_csv(path+"/pred.csv",sep="\t")
                else: df.to_csv(path+"/true.csv",sep="\t")

            
           
            #df = pd.DataFrame(data)


#data = np.load()