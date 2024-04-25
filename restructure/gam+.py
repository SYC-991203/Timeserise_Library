import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils.metrics import*
from torch.utils.data import TensorDataset, DataLoader, random_split


Device = "cuda" if torch.cuda.is_available() else "cpu"
lambda_value = 1e-2
exp_itme = "jn_4" ## 修改这个就行了 就修改成项目_IMF个数 其他什么都不用改了
type_names_list = ["jn","nd","ht"]

tensorset_dic={
    "jn_3":0,
    "jn_4":1,
    "jn_5":2,
    "jn_6":3,
    "nd_3":4,
    "nd_4":5,
    "nd_5":6,
    "ht_3":7,
    "ht_4":8,
    "ht_5":9,
    "ht_6":10,
}
type_key = exp_itme.split("_",1)[0]
imf_nums = int(exp_itme.split("_",1)[1])

input_dim = imf_nums+1
hidden_dim = 64
out_dim = input_dim

#在重构网络前定义RevIN层
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """Reversible Instance Normalization for Accurate Time-Series Forecasting
           against Distribution Shift, ICLR2021.

    Parameters
    ----------
    num_features: int, the number of features or channels.
    eps: float, a value added for numerical stability, default 1e-5.
    affine: bool, if True(default), RevIN has learnable affine parameters.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError('Only modes norm and denorm are supported.')
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = [0]
        self.mean = torch.median(x)
        # print(self.mean)
        # 计算每个数据点与均值之间的曼哈顿距离
        distances = torch.sum(torch.abs(x - self.mean), dim=dim2reduce)
        # 距离的均值，即为方差的估计值
        # print(distances)
        variance = torch.mean(distances)
        # print(variance)
        self.stdev = variance + self.eps

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

class SignalReconstructor(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(SignalReconstructor, self).__init__()

        # 定义归一化层
        self.revinlayer = RevIN(num_features=input_dim)

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv1d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm1d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv1d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, pred, imfs):
        # 将输入信号拼接
        data = torch.cat([pred] + imfs, dim=1)
        # 归一化
        x_norm = self.revinlayer(data, mode='norm')
        # x = self.revinlayer(x_norm, mode='denorm')

        x = x_norm.unsqueeze(-1).unsqueeze(-1)
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att
        x = x.squeeze(-1)

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        weights = torch.reshape(out, (hidden_dim, input_dim))
        weights = self.revinlayer(weights, mode='denorm')

        reconstructed_signal = weights[:, 0] * pred
        for i, imf in enumerate(imfs):
            reconstructed_signal += weights[:, i + 1] * imf
        # reconstructed_signal += weights[:,-1]*error

        return reconstructed_signal, weights


#数据解析
def get_all_data_from_loader(dataloader):
    all_data = list(zip(*[batch for batch in dataloader]))
    all_data = [torch.cat(data, dim=0) for data in all_data]
    return all_data
def load_data(type_name:str,num_imf:int,file_path="/home/qsmx/Data/DYG_sgc_pred.xlsx")->TensorDataset:
    try:
        df =pd.read_excel(file_path,sheet_name=f"{type_name}_{num_imf}")
        data_dic = {}
        data_dic["s_pred"] = torch.tensor(df[f"{type_name}_s_pred"].values).unsqueeze(1).to(torch.float32).to(Device)
        for i in range(num_imf):
            imf_column = f"sgc_{i}"
            if imf_column in df.columns:
                data_dic[imf_column] = torch.tensor(df[imf_column].values).unsqueeze(1).to(torch.float32).to(Device)
            else:
                raise ValueError(f"Column {imf_column} not found in {type_name} data.")
        # data_dic["error"] = torch.tensor(df["error"].values).unsqueeze(1).to(torch.float32).to(Device)
        data_dic[f"{type_name}_true"] =torch.tensor(df[f"{type_name}_true"].values).unsqueeze(1).to(torch.float32).to(Device)
        dataset = TensorDataset(*data_dic.values())
        return dataset
    except Exception as e:
        print(f"Error for {type_name}:{e}")
def process_batch(batch_pram,num_imf:int,device):
    batch = [x.to(device) for x in batch_pram]
    s_pred = batch[0]
    imf_list = batch[1:num_imf+1]
    # error = batch[num_imf+1]
    true = batch[num_imf+1]
    return s_pred,imf_list,true

# 定义最小loss
min_loss = float('inf')
best_weights = None
model = SignalReconstructor(in_channels=input_dim, out_channels=out_dim).to(device=Device)
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
        s_pred,imfs,true = process_batch(batch_pram=batch,num_imf=imf_nums,device=Device)
        optimizer.zero_grad()
        reconstructed_signal, weights = model(s_pred,imfs)
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
print(f"mae_re: {mae_re:.4f}, mse_re: {mse_re:.4f}, rmse_re:{rmse_re:.4f},mape_re:{mape_re:.4f},mspe_re:{mspe_re}")
