import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. 生成数据集
np.random.seed(42)
n_samples = 500
n_outliers = 20

# 正常数据点 - 正态分布
X_normal = np.random.normal(0, 1, size=(n_samples, 2))

# 异常数据点 - 故意偏离的点
X_outliers = np.random.normal(0, 10, size=(n_outliers, 2))

# 合并正常点和异常点
X = np.vstack([X_normal, X_outliers])

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 3. 异常检测：通过PCA的重构误差来判断
X_reconstructed = pca.inverse_transform(X_pca)
reconstruction_error = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)

# 设置一个阈值判断异常
threshold = np.percentile(reconstruction_error, 95)  # 设定95%分位为阈值
outliers = reconstruction_error > threshold

# 4. 画图
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# 图1: 原始数据的分布
axs[0].scatter(X_normal[:, 0], X_normal[:, 1], c='blue', label='Normal data')
axs[0].scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', label='Outliers')
axs[0].set_title('Original Data with Outliers')
axs[0].legend()

# 图2: PCA降维后的数据
axs[1].scatter(X_pca[:, 0], X_pca[:, 1], c=['red' if o else 'blue' for o in outliers], label='PCA-transformed data')
axs[1].set_title('PCA-transformed Data')
axs[1].legend()

# 图3: 重构误差与阈值
axs[2].scatter(range(len(reconstruction_error)), reconstruction_error, c=['red' if o else 'blue' for o in outliers])
axs[2].axhline(y=threshold, color='green', linestyle='--', label='Threshold')
axs[2].set_title('Reconstruction Error with Outliers Highlighted')
axs[2].legend()

# 美化和展示
for ax in axs:
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True)

plt.tight_layout()
plt.show()