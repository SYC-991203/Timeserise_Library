import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from utils.metrics import*
from torch.utils.data import TensorDataset, DataLoader, random_split


Device = "cuda" if torch.cuda.is_available() else "cpu"
lambda_value = 1e-2
exp_itme = "our_4" ## 修改这个就行了
type_names_list = ["cer","kla","our"]

tensorset_dic={
    "cer_1":0,
    "cer_2":1,
    "cer_3":2,
    "cer_4":3,
    "cer_5":4,
    "cer_6":5,
    "kla_2":5,
    "kla_3":6,
    "kla_4":7,
    "kla_5":8,
    "kla_6":9,
    "our_2":10,
    "our_3":11,
    "our_4":12,
    "our_5":13,
    "our_6":14,
}
type_key = exp_itme.split("_",1)[0]
imf_nums = int(exp_itme.split("_",1)[1])

input_dim = imf_nums+2
hidden_dim = 64
out_dim = input_dim


class SignalReconstructor(nn.Module):
    def __init__(self):
        super(SignalReconstructor, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()  # 从5个信号到5个权重
        self.fc2 = nn.Linear(hidden_dim,out_dim) # 确保输出的所有权重和为1

    def forward(self, pred, imfs,error,K):
        # 将输入信号拼接

        x = torch.cat([pred]+imfs+[error], dim=1)
        # 通过网络获取权重
        x = self.relu(self.fc1(x))
        weights = self.fc2(x)
        weights[:,0] = torch.sigmoid(weights[:,0])
        weights[:,-1] = torch.sigmoid(weights[:,-1])
        other_weights = F.softmax(weights[:,1:-1],dim = 1 )*K ## K是imf个数-1
        weights = torch.cat((weights[:, 0:1], other_weights, weights[:, -1:]), dim=1)     
        # 计算重构信号
        reconstructed_signal = weights[:, 0] * pred
        for i,imf in enumerate(imfs):
            reconstructed_signal += weights[:,i+1]*imf
        reconstructed_signal += weights[:,-1]*error

        return reconstructed_signal, weights
#数据解析    
def get_all_data_from_loader(dataloader):
    all_data = list(zip(*[batch for batch in dataloader]))
    all_data = [torch.cat(data, dim=0) for data in all_data]
    return all_data
def load_data(type_name:str,num_imf:int,file_path="./data/SignalRes/dyg_vmd_exp.xlsx")->TensorDataset:
    try:
        df =pd.read_excel(file_path,sheet_name=f"{type_name}_{num_imf}")
        data_dic = {}
        data_dic["s_pred"] = torch.tensor(df[f"{type_name}_s_pred"].values).unsqueeze(1).to(torch.float32).to(Device)
        for i in range(num_imf):
            imf_column = f"imf_{i}"
            if imf_column in df.columns:
                data_dic[imf_column] = torch.tensor(df[imf_column].values).unsqueeze(1).to(torch.float32).to(Device)
            else:
                raise ValueError(f"Column {imf_column} not found in {type_name} data.")
        data_dic["error"] = torch.tensor(df["error"].values).unsqueeze(1).to(torch.float32).to(Device)
        data_dic[f"{type_name}_true"] =torch.tensor(df[f"{type_name}_true"].values).unsqueeze(1).to(torch.float32).to(Device)   
        dataset = TensorDataset(*data_dic.values())
        return dataset
    except Exception as e:
        print(f"Error for {type_name}:{e}")
def process_batch(batch_pram,num_imf:int,device):
    batch = [x.to(device) for x in batch_pram]
    s_pred = batch[0]
    imf_list = batch[1:num_imf+1]
    error = batch[num_imf+1]
    true = batch[num_imf+2]
    return s_pred,imf_list,error,true

# 定义最小loss    
min_loss = float('inf')
best_weights = None
model = SignalReconstructor().to(device=Device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
mse_loss = nn.MSELoss()
data = load_data(type_key,imf_nums)
## 一口气把所有数据都读进来,后续可以一口气全跑完

# tensorset_list=[]
# for item in type_names_list:
#     for i in range(1,7): 
#         data = load_data(item,i)
#         if data:
#             print(f"Loaded data for {item}_{i}")
#             tensorset_list.append(data)

# for type_name,imf_nums in type_names.items():
#     data = load_data(type_name,imf_nums)
#     if data:
#         print(f"Loaded data for {type_name}_{imf_nums}")
#         tensorset_list.append(data)


## 数据读取没问题
# tensor_value = tensorset_dic.get(exp_itme,0)
# print(tensor_value)
dataset =data
# 划分为训练集和测试集
train_size = int(0.8 * len(dataset))  # 假设训练集占 80%
test_size = len(dataset) - train_size
## 顺序分割数据集
# train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
# test_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 训练过程
print(f"Training:Train item:{type_key},IMF num:{imf_nums}")
for epoch in range(200):  # 训练200轮
    model.train()
    for batch in train_loader:
        s_pred,imfs,error,true = process_batch(batch_pram=batch,num_imf=imf_nums,device=Device)
        optimizer.zero_grad()
        reconstructed_signal, weights = model(s_pred,imfs,error,K=imf_nums-1)
        #print(weights)
        loss = mse_loss(reconstructed_signal, true)
    # 可以在这里添加模型复杂度的惩罚项
    # 例如: loss += lambda * torch.sum(weights**2)
        loss.backward()
        optimizer.step()
        if loss.item() < min_loss:
            min_loss = loss.item()
            best_weights = weights.detach().clone()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
## 计算指标
weight = torch.mean(best_weights,dim = 0,keepdim=True)
print(f"Minimum Loss: {min_loss}, Best Weights: {best_weights},weight_mean:{weight}")

test_data_list = []

# 迭代 test_dataset 中的所有数据
for i in range(len(test_dataset)):
    single_sample = test_dataset[i]
    single_sample_concatenated = torch.cat(single_sample, dim=0)  # 假设每个样本是一个包含多个张量的元组
    test_data_list.append(single_sample_concatenated)

# 将列表中的所有张量堆叠成一个大张量
test_data_tensor = torch.stack(test_data_list)
test_feature = test_data_tensor[:,:-1]
test_true = test_data_tensor[:,-1].unsqueeze(1)
# 检查最终张量的形状
print(test_data_tensor.shape) 
## 人工调整权重值
# weight = [0,0.33,0.33,0.33,0]
# weight = torch.tensor(weight).view(1,input_dim).to(device=Device)
#concatenated_data = torch.cat((s_pred,kla_imf0, kla_imf1, kla_imf2, kla_error), dim=1)
re_signal = torch.matmul(test_feature,weight.t()).view(-1,1)

test_true = test_true.cpu().detach().numpy()
re_signal = re_signal.cpu().detach().numpy()
mae_re, mse_re, rmse_re, mape_re, mspe_re = metric(pred=re_signal, true=test_true)
print(f"Result:Train item:{type_key},IMF num:{imf_nums}")
print(f"mae_re: {mae_re:.4f}, mse_re: {mse_re:.4f},rmse_re:{rmse_re:.4f},mape_re:{mape_re:.4f},mspe_re:{mspe_re}")



