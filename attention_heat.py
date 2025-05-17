import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 文件路径
file_path = r"E:\ProgramFiles\论文KBS\weights.xlsx"

df = pd.read_excel(file_path, sheet_name='ht')

print(df)

spatial_att = df.iloc[0:20,0:6]

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = (5, 4)

plt.rcParams.update({
    'font.size': 10,            # 全局字体大小
    'axes.titlesize': 10,       # 图表标题字体大小
    'axes.labelsize': 10,       # 坐标轴标签字体大小
    'xtick.labelsize': 9,      # x轴刻度标签字体大小
    'ytick.labelsize': 9,      # y轴刻度标签字体大小
    'legend.fontsize': 9,      # 图例字体大小
    'figure.titlesize': 10      # 整个图的标题字体大小
})

# 使用 Seaborn 绘制热力图
plt.rc('font',family='Times New Roman')
sns.heatmap(spatial_att, cmap='GnBu', annot=False, cbar=True, vmin=0, vmax=1)
plt.xlabel("Channels")
plt.ylabel("Time steps")
plt.title(r"Attention Tensor of Sugar ($\mathcal{S}$)")
save_path = r"E:\ProgramFiles\论文KBS\FIGheatmap_ht.png"
plt.savefig(save_path, bbox_inches='tight')

plt.show()