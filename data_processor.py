import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from vmdpy import VMD
import pandas as pd
import os

cur_dir = os.getcwd() ##debug 路径和run路径不一样
print("cur_dir:",cur_dir)
# read dyg_pd
original_data = pd.read_csv("./data/DYG/DYG_clean.csv",header=0)# 2900行 18列
print(original_data.columns)
our = original_data['our']
cer = original_data['cer']
kla = original_data['kla']
# -----测试信号及其参数--start-------------


f_1=100;f_2=200;f_3=300
data_list = [our,cer,kla]
# v_1=(np.cos(2*np.pi*f_1*t));v_2=1/4*(np.cos(2*np.pi*f_2*t));v_3=1/16*(np.cos(2*np.pi*f_3*t)) #v的三个基础信号
# v=[v_1,v_2,v_3] # 测试信号所包含的各成分
# f=v_1+v_2+v_3+0.1*np.random.randn(v_1.size)  # 测试信号，最终的合成信号
# -----测试信号及其参数--end----------
# alpha 惩罚系数；带宽限制经验取值为抽样点长度1.5-2.0倍.
# 惩罚系数越小，各IMF分量的带宽越大，过大的带宽会使得某些分量包含其他分量言号;
# a值越大，各IMF分量的带宽越小，过小的带宽是使得被分解的信号中某些信号丢失该系数常见取值范围为1000~3000
alpha=4000
tau=0 # tau 噪声容限，即允许重构后的信号与原始信号有差别。
K=3# K 分解模态（IMF）个数
DC=0 # DC 若为0则让第一个IMF为直流分量/趋势向量
init=1 # init 指每个IMF的中心频率进行初始化。当初始化为1时，进行均匀初始化。
tol=1e-7 # 控制误差大小常量，决定精度与迭代次数
# 输出U是各个IMF分量，u_hat是各IMF的频谱，omega为各IMF的中心频率
result_u_list =[]
error_our_list = []
error_cer_list = []
error_kla_list = []

u_our,u_our_hat,omega_our = VMD(our,alpha, tau, K, DC, init, tol) ## tuple:u, u_hat, omega
u_cer,u_cer_hat,omege_cer = VMD(cer,alpha, tau, K, DC, init, tol)
u_kla,u_kla_hat,omega_kla = VMD(kla,alpha, tau, K, DC, init, tol)
print(len(our))
print(len(cer))
print(len(kla))
print(len(u_our))
Fs=u_our.shape[1] # 采样频率
N=u_our.shape[1] # 采样点数
t=np.arange(1,N+1)/N
print(len(t))
fre_axis=np.linspace(0,Fs/2,int(N/2))
for i in range(u_our.shape[1]):
    e_our = our[i]
    e_cer = cer[i]
    e_kla = kla[i]
    for j in range(K):
        e_our -= u_our[j][i]
        e_cer -= u_cer[j][i] 
        e_kla -= u_kla[j][i]
    error_our_list.append(e_our)
    error_cer_list.append(e_cer)
    error_kla_list.append(e_kla)

result_u_list.extend([u_our.T,u_cer.T,u_kla.T])
u_data = {}
for i in range(len(u_our)):
    u_data[f"u_our_imf{i}"] = u_our[i]
    u_data[f"u_cer_imf{i}"] = u_cer[i]
    u_data[f"u_kla_imf{i}"] = u_kla[i]
u_data.update({"u_our_imf_error":error_our_list,"u_cer_imf_error":error_cer_list,"u_kla_imf_error":error_kla_list})

# u_data = {"u_our_imf0":u_our[0],"u_our_imf1":u_our[1],"u_our_imf2":u_our[2],
#         "u_cer_imf0":u_cer[0],"u_cer_imf1":u_cer[1],"u_cer_imf2":u_cer[2],
#         "u_kla_imf0":u_kla[0],"u_kla_imf1":u_kla[1],"u_kla_imf2":u_kla[2]}
u_df = pd.DataFrame(u_data)
# existing_cols = original_data.columns.intersection(u_df.columns)
# if len(existing_cols)>0:
#     original_data[existing_cols] = u_df[existing_cols]
# else:## 没有写入。直接续写
original_with_u = pd.concat([original_data,u_df],axis =1)
original_with_u.to_csv(f"./data/DYG/DYG_vmd_{K}.csv",index=False)
print(u_df.shape)

    

u, u_hat, omega = VMD(our,alpha, tau, K, DC, init, tol) #单纯画图可以从这个修改变量
print(u.shape)
# 分解出的变量颜色为 蓝 橙 绿 imf为蓝色 imf2为橙色 imf3为绿色
# 1 画原始信号our cer kla
plt.figure(figsize=(10,7));plt.subplot(K+1, 1, 1);plt.plot(t,our)
for i,y in enumerate(data_list):
    plt.subplot(K+1, 1, i+2);plt.plot(t,y)
plt.suptitle('Original input signal with our cer and kla');plt.show()

# 2 分解出来的各IMF分量
plt.figure(figsize=(10,7))
plt.plot(t,u.T);plt.title('all Decomposed modes');plt.show()  # u.T是对u的转置

# 3 各IMF分量的fft幅频图
plt.figure(figsize=(10, 7), dpi=80)
for i in range(K):
    plt.subplot(K, 1, i + 1)
    fft_res=np.fft.fft(u[i, :])
    plt.plot(fre_axis,abs(fft_res[:int(N/2)])/(N/2))
    plt.title('(FFT) amplitude frequency of IMF {}'.format(i + 1))
plt.show()

# 4 分解出来的各IMF分量的频谱
# print(u_hat.shape,t.shape,omega.shape)
plt.figure(figsize=(10, 7), dpi=80)
for i in range(K):
    plt.subplot(K, 1, i + 1)
    plt.plot(fre_axis,abs(u_hat[:, i][int(N/2):])/(N/2))
    plt.title('(VMD)amplitude frequency of the modes{}'.format(i + 1))
plt.tight_layout();plt.show()

# 5 各IMF的中心频率
plt.figure(figsize=(12, 7), dpi=80)
for i in range(K):
    plt.subplot(K, 1, i + 1)
    plt.plot(omega[:,i]) # X轴为迭代次数，y轴为中心频率
    plt.title('mode center-frequencies{}'.format(i + 1))
plt.tight_layout();plt.show()

plt.figure(figsize=(10,7))
plt.plot(t,np.sum(u,axis=0))
plt.title('reconstructed signal')
