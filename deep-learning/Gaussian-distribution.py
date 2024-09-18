import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# 高斯分布算法
# https://mp.weixin.qq.com/s/-a8q2C4DH1JUzg2aRmymVg
# 1. 生成虚拟数据集
np.random.seed(42)
X, y_true = make_blobs(n_samples=1000, centers=2, cluster_std=0.60, random_state=42)

# 2. 添加一些异常点
X_outliers = np.random.uniform(low=-10, high=10, size=(20, 2))
X = np.vstack([X, X_outliers])

# 3. 使用高斯混合模型进行拟合
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(X)

# 4. 预测每个点属于哪个簇
y_pred = gmm.predict(X)

# 5. 计算每个点的异常分数（基于概率）
scores = gmm.score_samples(X)

# 6. 设置异常分数阈值（可以调整）
threshold = np.percentile(scores, 5)  # 设定5%的数据为异常点
outliers = scores < threshold

# 7. 画图
plt.figure(figsize=(14, 7))

# (1) 数据分布图
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50, label='Clusters')
plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', s=100, edgecolor='k', label='Outliers')
plt.title('Data Distribution with GMM Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# (2) 异常点检测图
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c='blue', s=50, label='Data points')
plt.scatter(X[outliers, 0], X[outliers, 1], c='red', s=100, edgecolor='k', label='Detected Outliers')
plt.title('Anomaly Detection with GMM')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.show()