import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# 局部异常因子检测
# https://mp.weixin.qq.com/s/-a8q2C4DH1JUzg2aRmymVg
# 设置随机种子
np.random.seed(42)

# 生成虚拟数据集
n_samples = 1000
X_inliers = 0.3 * np.random.randn(n_samples // 2, 2) + [2, 2]  # 正常数据
X_outliers = np.random.uniform(low=-4, high=6, size=(n_samples // 4, 2))  # 异常数据
X = np.concatenate([X_inliers, X_outliers], axis=0)  # 总数据集

# 使用局部异常因子(LOF)进行异常检测
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = lof.fit_predict(X)
lof_scores = -lof.negative_outlier_factor_  # LOF 值（负值的相反数）

# 绘制图像
plt.figure(figsize=(12, 8))

# 1. 绘制原始数据集的散点图，标注异常点和正常点
plt.subplot(2, 2, 1)
plt.title("Data Points and Outliers", fontsize=15)
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], color='blue', edgecolor='k', s=50, label='Inliers')
plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], color='red', edgecolor='k', s=50, label='Outliers')
plt.legend(loc='upper left')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# 2. 绘制 LOF 值的图，颜色表示LOF的大小
plt.subplot(2, 2, 2)
plt.title("LOF Values", fontsize=15)
colors = np.where(lof_scores > np.percentile(lof_scores, 90), 'red', 'blue')  # 高于90百分位的为红色
plt.scatter(X[:, 0], X[:, 1], c=colors, s=50, cmap='rainbow', edgecolor='k')
plt.colorbar(label='LOF Score (Higher -> More Outlier)')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# 3. 绘制 LOF 分布图
plt.subplot(2, 2, 3)
plt.title("LOF Score Distribution", fontsize=15)
plt.hist(lof_scores, bins=20, color='orange', edgecolor='black')
plt.axvline(np.percentile(lof_scores, 90), color='red', linestyle='--', label='90th Percentile')
plt.legend()
plt.xlabel("LOF Score")
plt.ylabel("Frequency")

# 4. 绘制前5个最异常的数据点，突出显示异常点的散点图
plt.subplot(2, 2, 4)
plt.title("Top 5 Outliers Highlighted", fontsize=15)
top_5_outliers = np.argsort(lof_scores)[-5:]  # 找到前5个最异常点
plt.scatter(X[:, 0], X[:, 1], color='blue', edgecolor='k', s=50, label='Data Points')
plt.scatter(X[top_5_outliers, 0], X[top_5_outliers, 1], color='lime', edgecolor='k', s=100, label='Top 5 Outliers', marker='x')
plt.legend(loc='upper left')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# 展示图像
plt.tight_layout()
plt.show()