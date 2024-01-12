import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from utils.metrics import*
from torch.utils.data import TensorDataset, DataLoader, random_split

input_dim = 5
hidden_dim = 64
out_dim = 5
Device = "cuda" if torch.cuda.is_available() else "cpu"
lambda_value = 1e-2

class SignalReconstructor(nn.Module):
    def __init__(self):
        super(SignalReconstructor, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()  # 从5个信号到5个权重
        self.fc2 = nn.Linear(hidden_dim,out_dim) # 确保输出的所有权重和为1

    def forward(self, pred, imf0, imf1,imf2,error):
        # 将输入信号拼接

        x = torch.cat((pred, imf0, imf1, imf2, error), dim=1)
        # 通过网络获取权重
        x = self.relu(self.fc1(x))
        weights = self.fc2(x)
        weights[:,0] = torch.sigmoid(weights[:,0])
        weights[:,-1] = torch.sigmoid(weights[:,-1])
        other_weights = F.softmax(weights[:,1:-1],dim = 1 )*2
        weights_pred = weights[:, 0].unsqueeze(1)
        weights_error = weights[:, -1].unsqueeze(1)
        weights = torch.cat((weights_pred, other_weights, weights_error), dim=1)
        # 计算重构信号
        reconstructed_signal = weights[:, 0] * pred + weights[:, 1] * imf0 + \
        weights[:,2]*imf1 + weights[:,3]*imf2 + weights[:,4]*error

        return reconstructed_signal, weights
    
def get_all_data_from_loader(dataloader):
    all_data = list(zip(*[batch for batch in dataloader]))
    all_data = [torch.cat(data, dim=0) for data in all_data]
    return all_data
# 定义最小loss    
min_loss = float('inf')
best_weights = None
# 创建模型
model = SignalReconstructor().to(device=Device)

# 优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
mse_loss = nn.MSELoss()
# 准备数据
kla_df = pd.read_excel("./data/SignalRes/dyg_vmd_exp.xlsx",sheet_name="kla_3")
s_pred = torch.tensor(kla_df["S_PRED"].values).unsqueeze(1).to(torch.float32).to(Device)## 把sigel pred的原始信号也输入进去
kla_imf0 = torch.tensor(kla_df["imf_0"].values).unsqueeze(1).to(torch.float32).to(Device)
kla_imf1 = torch.tensor(kla_df["imf_1"].values).unsqueeze(1).to(torch.float32).to(Device)
kla_imf2 = torch.tensor(kla_df["imf_2"].values).unsqueeze(1).to(torch.float32).to(Device)
kla_error  = torch.tensor(kla_df["error"].values).unsqueeze(1).to(torch.float32).to(Device) ## 残差数据也输入进去
kla_true = torch.tensor(kla_df["KLA_TRUE"].values).unsqueeze(1).to(torch.float32).to(Device)

dataset = TensorDataset(s_pred, kla_imf0, kla_imf1, kla_imf2, kla_error, kla_true)

# 划分为训练集和测试集
train_size = int(0.8 * len(dataset))  # 假设训练集占 80%
test_size = len(dataset) - train_size
## 顺序分割数据集
# train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
# test_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 训练过程
for epoch in range(200):  # 训练200轮
    model.train()
    for batch in train_loader:
        s_pred,kla_imf0,kla_imf1,kla_imf2,kla_error,kla_true = [x.to(device = Device)for x in batch]
        optimizer.zero_grad()
        reconstructed_signal, weights = model(s_pred, kla_imf0, kla_imf1,kla_imf2,kla_error)
        #print(weights)
        loss = mse_loss(reconstructed_signal, kla_true)
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
    # 获取单个样本
    single_sample = test_dataset[i]
    
    # 将单个样本中的所有张量拼接成一个张量
    single_sample_concatenated = torch.cat(single_sample, dim=0)  # 假设每个样本是一个包含多个张量的元组
    
    # 添加到列表中
    test_data_list.append(single_sample_concatenated)

# 将列表中的所有张量堆叠成一个大张量
test_data_tensor = torch.stack(test_data_list)
test_feature = test_data_tensor[:,:-1]
test_true = test_data_tensor[:,-1].unsqueeze(1)
# 检查最终张量的形状
print(test_data_tensor.shape) 
## 人工调整权重值
# weight = [0,0.33,0.33,0.33,0]
# weight = torch.tensor(weight).view(1,5).to(device=Device)
#concatenated_data = torch.cat((s_pred,kla_imf0, kla_imf1, kla_imf2, kla_error), dim=1)
re_signal = torch.matmul(test_feature,weight.t()).view(-1,1)

test_true = test_true.cpu().detach().numpy()
re_signal = re_signal.cpu().detach().numpy()
mae_re, mse_re, rmse_re, mape_re, mspe_re = metric(pred=re_signal, true=test_true)

print(f"mae_re: {mae_re}, mse_re: {mse_re},rmse_re:{rmse_re},mape_re:{mape_re},mspe_re:{mspe_re}")



