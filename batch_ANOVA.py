import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = (6, 4)

plt.rcParams.update({
    'font.size': 10,            # 全局字体大小
    'axes.titlesize': 10,       # 图表标题字体大小
    'axes.labelsize': 8,       # 坐标轴标签字体大小
    'xtick.labelsize': 8,      # x轴刻度标签字体大小
    'ytick.labelsize': 8,      # y轴刻度标签字体大小
    'legend.fontsize': 10,      # 图例字体大小
    'figure.titlesize': 10      # 整个图的标题字体大小
})

# 读取数据
original_data = pd.read_csv(r"E:\ProgramFiles\DYG_data\DYG_new.csv", header=0)

# 提取 'jn', 'nd', 'ht' 列
jn = original_data['jn']
nd = original_data['nd']
ht = original_data['ht']

# 每100个采样点划分一个batch
batch_size = 100

# 计算批次数
num_batches = len(jn) // batch_size

# 使用 numpy 的 array_split 方法将数据划分成batch
batches = np.array_split(jn, num_batches)

# 对每个batch计算均值和方差
means = []
variances = []

for batch in batches:
    mean = np.mean(batch)  # 计算均值
    variance = np.var(batch)  # 计算方差
    means.append(mean)
    variances.append(variance)

# 计算方差的均值
mean_variance = np.mean(variances)

# 计算每个批次的方差与均值的差值
batch_diff = [(i+1, abs(variance - mean_variance)) for i, variance in enumerate(variances)]

# 按照差值排序，选择差值最小的前10个批次
sorted_batches = sorted(batch_diff, key=lambda x: x[1])

# 提取最接近均值方差的前 10 个批次的索引
top_10_batches = [batches[batch[0]-1] for batch in sorted_batches[:10]]  # batch[0]-1 是因为索引从 0 开始

# 绘制10个批次的箱线图
plt.boxplot(top_10_batches, labels=[f'Batch_{batch[0]}' for batch in sorted_batches[:10]])
plt.title("Concentration ($\mathcal{C}$) Variance Distribution Across Different Batches")
plt.ylabel("Concentration")
plt.xlabel("Different Batches")
# 设置y轴的范围（假设我们设定范围为0到1）
# plt.ylim(0.3, 2)  # 这里你可以根据实际情况调整上下限

# 保存图像为 jpg 文件
# plt.savefig(r"E:\ProgramFiles\论文BioPharmaSoftNet\图\top_10_boxplot.jpg", dpi=600, bbox_inches='tight')

plt.show()