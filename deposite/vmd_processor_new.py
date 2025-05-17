from vmdpy import VMD
import matplotlib.pyplot as plt
import numpy as np
from PyEMD import Visualisation
from scipy.signal import hilbert
import pandas as pd

#求窄带信号的边际谱
def mspect(Fs,signal,draw=1):
    fmin,fmax=0,Fs/2
    size=len(signal)//2
    df=(fmax-fmin)/(size-1)
    t=np.arange(0,len(signal)/Fs,1/Fs)
    vis = Visualisation()
    #希尔伯特变化
    signal=signal.reshape(1,-1)
    #求瞬时频率
    freqs = abs(vis._calc_inst_freq(signal, t, order=False, alpha=None))
    #求瞬时幅值
    amp= abs(hilbert(signal))
    #去掉为1的维度
    freqs=np.squeeze(freqs)
    amp=np.squeeze(amp)
    result=np.zeros(size)
    for i,j in zip(freqs,amp):
        if i>=fmin and i<=fmax:
            result[round((i-fmin)/df)]+=j
    
    f=np.arange(fmin,size*df,df)
    #可视化
    if draw==1:                           #可视化
        plt.figure()
        plt.rcParams['font.sans-serif']='Times New Roman'
        plt.plot(f,result)
        plt.xlabel('f/HZ',fontsize=16)
        plt.ylabel('amplitude',fontsize=16)
        plt.title('Marginal Spectrum',fontsize=20)
    
    return f,result


#基于稀疏指标自适应确定K值的VMD分解   
def Auto_VMD_main(signal,Fs,draw=1,maxK=10):
    
    #vmd参数设置
    alpha = 3000       # moderate bandwidth constraint   2000
    tau = 0.            # noise-tolerance (no strict fidelity enforcement)
    DC = 0             # no DC part imposed
    init = 1           # initialize omegas uniformly
    tol = 1e-7
    
    #寻找最佳K
    S=[[],[]]
    flag,idx=-2,2
    for K in range(2,maxK+1):
        IMFs,_,_=VMD(signal, alpha, tau, K, DC, init, tol)                                    #分解信号
        M_spect=[]
        max_M=[]
        for i in range(len(IMFs)):
            # _,_=fftlw(Fs,IMFs[i,:],1)
            _,M=mspect(Fs,IMFs[i,:],0)
            max_M.append(max(M))
            temp=np.mean(M**2)/(np.mean(M)**2)
            M_spect.append(temp)
        
        max_M=max_M/max(max_M)
        S_index=np.mean(max_M*M_spect)
        if S_index>flag:
            flag=S_index
            idx=K
        S[0].append(K)
        S[1].append(S_index)
 
    
    #用最佳K值分解信号
    IMFs, _, _ = VMD(signal, alpha, tau, idx, DC, init, tol)
    print(f"signal:{signal.name},idx:{idx}")
    #可视化寻优过程与最终结果
    if draw==1:
        plt.figure()
        plt.rcParams['font.sans-serif']='Times New Roman'
        plt.plot(S[0],S[1])
        plt.scatter([idx],[flag],c='r',marker='*')
        plt.xlabel('K',fontsize=16)
        plt.ylabel('Sparse index',fontsize=16)
        plt.title('Optimization Process',fontsize=20)
        
        plt.figure()
        for i in range(len(IMFs)):
            plt.subplot(len(IMFs),1,i+1)
            plt.plot(t,IMFs[i])
            if i==0:
                plt.rcParams['font.sans-serif']='Times New Roman'
                plt.title('Decomposition Signal',fontsize=14)
            elif i==len(IMFs)-1:
                plt.rcParams['font.sans-serif']='Times New Roman'
                plt.xlabel('Time/s')
    return IMFs,idx
def hhtlw(IMFs,t,f_range=[0,500],t_range=[0,1],ft_size=[128,128],draw=1):
    fmin,fmax=f_range[0],f_range[1]         #时频图所展示的频率范围
    tmin,tmax=t_range[0],t_range[1]         #时间范围
    fdim,tdim=ft_size[0],ft_size[1]         #时频图的尺寸（分辨率）
    dt=(tmax-tmin)/(tdim-1)
    df=(fmax-fmin)/(fdim-1)
    vis = Visualisation()
    #希尔伯特变化
    c_matrix=np.zeros((fdim,tdim))
    for imf in IMFs:
        imf=np.array([imf])
        #求瞬时频率
        freqs = abs(vis._calc_inst_freq(imf, t, order=False, alpha=None))
        #求瞬时幅值
        amp= abs(hilbert(imf))
        #去掉为1的维度
        freqs=np.squeeze(freqs)
        amp=np.squeeze(amp)
        #转换成矩阵
        temp_matrix=np.zeros((fdim,tdim))
        n_matrix=np.zeros((fdim,tdim))
        for i,j,k in zip(t,freqs,amp):
            if i>=tmin and i<=tmax and j>=fmin and j<=fmax:
                temp_matrix[round((j-fmin)/df)][round((i-tmin)/dt)]+=k
                n_matrix[round((j-fmin)/df)][round((i-tmin)/dt)]+=1
        n_matrix=n_matrix.reshape(-1)
        idx=np.where(n_matrix==0)[0]
        n_matrix[idx]=1
        n_matrix=n_matrix.reshape(fdim,tdim)
        temp_matrix=temp_matrix/n_matrix
        c_matrix+=temp_matrix
    
    t=np.linspace(tmin,tmax,tdim)
    f=np.linspace(fmin,fmax,fdim)
    #可视化
    if draw==1:
        plt.ion()
        fig,axes=plt.subplots()
        plt.rcParams['font.sans-serif']='Times New Roman'
        plt.contourf(t, f, c_matrix,cmap="jet")
        plt.xlabel('Time/s',fontsize=16)
        plt.ylabel('Frequency/Hz',fontsize=16)
        plt.title('Hilbert spectrum',fontsize=20)
        x_labels=axes.get_xticklabels()
        [label.set_fontname('Times New Roman') for label in x_labels]
        y_labels=axes.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in y_labels]
        
        plt.show()
    return t,f,c_matrix


if __name__ == '__main__':
    # 指定文件路径和工作表名称
    InFile = r"E:\ProgramFiles\conference\DYG_data\DYG_data.xlsx"
    sheet_name = 'Sheet3'  # 替换为你想要的工作表名称

    # 读取 Excel 文件中的指定 Sheet
    original_data = pd.read_excel(InFile, sheet_name=sheet_name)

    # 设置采样频率和采样点数
    Fs = 9882  # 采样频率
    N = 9882  # 采样点数
    t = np.arange(1, N + 1) / N

    # 定义变量列表
    variables = ['jn', 'nd', 'hx']

    # 创建一个字典来存储所有变量的结果和误差
    all_u_data = {}

    # 处理每个变量
    for var in variables:
        # 提取变量的数据
        data = original_data[var]

        # 定义结果和误差列表
        result_u_list = []
        error_list = []

        # 使用 Auto_VMD_main 函数处理变量
        u_var, k_var = Auto_VMD_main(data, Fs, draw=1, maxK=10)

        # 计算误差并存储在列表中
        for i in range(len(data)):
            e_var = data[i]
            for j in range(k_var):
                e_var -= u_var[j][i]
            error_list.append(e_var)

        # 将处理后的数据存储在字典中
        for i in range(len(u_var)):
            all_u_data[f"u_{var}_imf{i}"] = u_var[i]
        all_u_data[f"u_{var}_imferror"] = error_list

    # 将所有处理后的数据转换为 DataFrame
    u_df = pd.DataFrame(all_u_data)

    # 将处理后的数据与原始数据合并并保存到 CSV 文件中
    original_with_u = pd.concat([original_data, u_df], axis=1)
    output_file = r"E:\ProgramFiles\conference\DYG_data\DYG_vmd_combined.csv"
    original_with_u.to_csv(output_file, index=False)

    # 打印处理后的 DataFrame 形状
    print(u_df.shape)
