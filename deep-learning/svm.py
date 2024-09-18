import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# https://mp.weixin.qq.com/s/-a8q2C4DH1JUzg2aRmymVg
# 支持向量机
# 设置随机种子以便复现
np.random.seed(42)

# 生成正常数据 (两簇高斯分布)
X_inliers = 0.3 * np.random.randn(100, 2)
X_inliers = np.r_[X_inliers + 2, X_inliers - 2]

# 生成异常数据
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# 数据集：包含正常点和异常点
X = np.r_[X_inliers, X_outliers]

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 训练One-Class SVM
svm = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.05)
svm.fit(X_scaled)

# 预测数据点是否是异常
y_pred = svm.predict(X_scaled)

# 找出异常点和正常点
X_normal = X[y_pred == 1]
X_anomalies = X[y_pred == -1]

# 绘制决策边界
xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
Z = svm.decision_function(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

# 创建一个图形窗口
plt.figure(figsize=(12, 8))

# 1. 绘制原始数据的分布
plt.subplot(221)
plt.scatter(X_inliers[:, 0], X_inliers[:, 1], c='blue', label='Inliers', s=40)
plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', label='Outliers', s=40)
plt.title('Original Data Distribution', fontsize=14)
plt.legend(loc="best")
plt.grid(True)

# 2. 绘制SVM的决策边界和支持向量
plt.subplot(222)
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu_r)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
plt.scatter(X_normal[:, 0], X_normal[:, 1], c='green', label='Normal Points', s=40)
plt.scatter(X_anomalies[:, 0], X_anomalies[:, 1], c='magenta', label='Anomalies', s=40)
plt.title('SVM Decision Boundary and Anomalies', fontsize=14)
plt.legend(loc="best")
plt.grid(True)

# 3. 绘制异常点的分布
plt.subplot(223)
plt.scatter(X_anomalies[:, 0], X_anomalies[:, 1], c='magenta', s=40)
plt.title('Detected Anomalies', fontsize=14)
plt.grid(True)

# 4. 绘制正常点的分布
plt.subplot(224)
plt.scatter(X_normal[:, 0], X_normal[:, 1], c='green', s=40)
plt.title('Detected Normal Points', fontsize=14)
plt.grid(True)

# 设置整体图形标题和布局
plt.suptitle('One-Class SVM for Anomaly Detection', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# 显示图形
plt.show()