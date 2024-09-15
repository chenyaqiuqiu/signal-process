import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import seaborn as sns

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 1. 生成虚拟数据集
# 创建三个不同的聚类
cluster_1 = np.random.normal(loc=(0, 0), scale=1.0, size=(150, 2))
cluster_2 = np.random.normal(loc=(5, 5), scale=1.0, size=(150, 2))
cluster_3 = np.random.normal(loc=(0, 5), scale=1.0, size=(150, 2))

# 添加异常点
anomalies = np.random.uniform(low=-5, high=10, size=(30, 2))

# 合并所有数据点
X = np.vstack((cluster_1, cluster_2, cluster_3, anomalies))

# 2. 应用 KNN 进行异常检测
k = 5  # 设置最近邻的数量
nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X)
distances, indices = nbrs.kneighbors(X)

# 计算每个点到其 k 个最近邻的平均距离作为异常得分
anomaly_scores = distances.mean(axis=1)

# 3. 数据可视化
plt.figure(figsize=(16, 8))

# 图形1：散点图，颜色表示异常得分
plt.subplot(1, 2, 1)
scatter = plt.scatter(X[:, 0], X[:, 1], c=anomaly_scores, cmap='plasma', edgecolor='k', alpha=0.7)
plt.colorbar(scatter, label='Anomaly Score')
plt.title('KNN Anomaly Detection Scatter Plot')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 标出异常得分较高的前1%的点
threshold = np.percentile(anomaly_scores, 99)
high_anomalies = X[anomaly_scores >= threshold]
plt.scatter(high_anomalies[:, 0], high_anomalies[:, 1], facecolors='none', edgecolors='r', s=100, label='High Anomalies')
plt.legend()

# 图形2：异常得分的分布直方图
plt.subplot(1, 2, 2)
sns.histplot(anomaly_scores, bins=50, color='teal', kde=True)
plt.axvline(threshold, color='r', linestyle='--', label='99% Threshold')
plt.title('Anomaly Score Distribution')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()