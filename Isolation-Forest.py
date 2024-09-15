import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import IsolationForest

# Step 1: 生成虚拟数据集
np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, flip_y=0.01, class_sep=1.5)

# 引入一些异常点
n_outliers = 50
X_outliers = np.random.uniform(low=-4, high=4, size=(n_outliers, 2))
X = np.vstack((X, X_outliers))
y_outliers = np.ones(n_outliers) * -1  # 异常点标签为-1
y = np.concatenate([y, y_outliers])

# Step 2: 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: 使用 Isolation Forest 进行异常检测
iso_forest = IsolationForest(contamination=0.05, random_state=42)
y_pred_train = iso_forest.fit_predict(X_train)
y_pred_test = iso_forest.predict(X_test)

# Step 4: 使用 Random Forest 进行分类训练
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 预测测试集
y_pred_rf = clf.predict(X_test)

# Step 5: 绘制数据图形
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# 图形1：原始数据集及异常点标记
colors = np.array(['blue', 'green'])
ax[0].scatter(X_train[:, 0], X_train[:, 1], c=colors[(y_pred_train > 0).astype(int)], marker='o', label='Normal Points')
ax[0].scatter(X_test[:, 0], X_test[:, 1], c='red', marker='x', label='Anomalies')
ax[0].set_title('Original Dataset with Anomalies', fontsize=15)
ax[0].set_xlabel('Feature 1', fontsize=12)
ax[0].set_ylabel('Feature 2', fontsize=12)
ax[0].legend(loc='best')

# 图形2：特征重要性
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
ax[1].bar(range(X.shape[1]), importances[indices], color='orange', align='center')
ax[1].set_xticks(range(X.shape[1]))
ax[1].set_xticklabels([f'Feature {i+1}' for i in indices])
ax[1].set_title('Random Forest Feature Importance', fontsize=15)
ax[1].set_xlabel('Features', fontsize=12)
ax[1].set_ylabel('Importance Score', fontsize=12)

# 调整布局
plt.tight_layout()
plt.show()