import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor

# 生成虚拟数据集 (时间序列数据)
np.random.seed(42)
n_samples = 500
time = np.arange(n_samples)
data = np.sin(0.1 * time) + np.random.normal(0, 0.1, n_samples)  # 正常时间序列数据

# 添加一些异常点
outliers_fraction = 0.05
n_outliers = int(outliers_fraction * n_samples)
outlier_indices = np.random.choice(time, n_outliers, replace=False)
data[outlier_indices] += np.random.uniform(3, 6, size=n_outliers)  # 异常点偏离正常范围

# 数据放入DataFrame中
df = pd.DataFrame({'time': time, 'value': data})

# 使用DBSCAN和LOF进行聚合增强型异常检测
dbscan = DBSCAN(eps=0.3, min_samples=5)
lof = LocalOutlierFactor(n_neighbors=20, contamination=outliers_fraction)

# DBSCAN聚类结果 (-1表示噪声点/异常)
dbscan_labels = dbscan.fit_predict(df[['value']])

# LOF结果 (-1表示异常)
lof_labels = lof.fit_predict(df[['value']])

# 将聚合结果整合到DataFrame
df['dbscan_label'] = dbscan_labels
df['lof_label'] = lof_labels
df['anomaly'] = ((df['dbscan_label'] == -1) | (df['lof_label'] == -1)).astype(int)

# 绘制图形：包括原始数据、异常点、聚类结果
plt.figure(figsize=(12, 8))

# 图1：时间序列数据及检测出的异常点
plt.subplot(3, 1, 1)
plt.plot(df['time'], df['value'], color='blue', label='Time Series Data')
plt.scatter(df['time'][df['anomaly'] == 1], df['value'][df['anomaly'] == 1], color='red', label='Anomalies', marker='x')
plt.title('Time Series with Detected Anomalies')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

# 图2：DBSCAN的聚类结果
plt.subplot(3, 1, 2)
colors = np.array(['blue', 'green', 'red'])
unique_labels = np.unique(dbscan_labels)
for label in unique_labels:
    plt.scatter(df['time'][df['dbscan_label'] == label], df['value'][df['dbscan_label'] == label],
                color=colors[label] if label != -1 else 'red', label=f'Cluster {label}' if label != -1 else 'Noise')
plt.title('DBSCAN Clustering Results')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

# 图3：LOF异常检测结果
plt.subplot(3, 1, 3)
plt.plot(df['time'], df['value'], color='blue', label='Time Series Data')
plt.scatter(df['time'][df['lof_label'] == -1], df['value'][df['lof_label'] == -1], color='orange', label='LOF Anomalies', marker='x')
plt.title('LOF Anomaly Detection Results')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

# 调整图像布局并显示
plt.tight_layout()
plt.show()