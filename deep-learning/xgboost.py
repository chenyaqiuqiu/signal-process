import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据集
data = pd.read_csv('house_prices.csv')  # 请确保你已经下载了Kaggle数据集

# 选择一些特征并处理缺失值
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']
X = data[features].fillna(0).values
y = data['SalePrice'].values

# 定义损失函数的梯度和Hessian
def gradient(y_true, y_pred):
    return y_pred - y_true

def hessian(y_true, y_pred):
    return np.ones_like(y_true)

# 构建决策树类
class XGBoostTree:
    def __init__(self, max_depth=3, lambda_reg=1):
        self.max_depth = max_depth
        self.lambda_reg = lambda_reg
        self.tree = {}

    def fit(self, X, g, h):
        self.tree = self._build_tree(X, g, h, depth=0)

    def _build_tree(self, X, g, h, depth):
        if depth == self.max_depth or len(X) <= 1:
            weight = -np.sum(g) / (np.sum(h) + self.lambda_reg)
            return weight

        best_gain = 0
        best_split = None

        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:, feature]):
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold

                if len(X[left_idx]) == 0 or len(X[right_idx]) == 0:
                    continue

                G_L, G_R = np.sum(g[left_idx]), np.sum(g[right_idx])
                H_L, H_R = np.sum(h[left_idx]), np.sum(h[right_idx])

                gain = 0.5 * ((G_L**2 / (H_L + self.lambda_reg)) +
                              (G_R**2 / (H_R + self.lambda_reg)) -
                              ((G_L + G_R)**2 / (H_L + H_R + self.lambda_reg)))

                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, threshold)

        if best_split is None:
            weight = -np.sum(g) / (np.sum(h) + self.lambda_reg)
            return weight

        feature, threshold = best_split
        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold

        return {
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(X[left_idx], g[left_idx], h[left_idx], depth + 1),
            'right': self._build_tree(X[right_idx], g[right_idx], h[right_idx], depth + 1)
        }

    def predict(self, X):
        return np.array([self._predict_one(row, self.tree) for row in X])

    def _predict_one(self, x, tree):
        if isinstance(tree, dict):
            feature = tree['feature']
            threshold = tree['threshold']
            if x[feature] <= threshold:
                return self._predict_one(x, tree['left'])
            else:
                return self._predict_one(x, tree['right'])
        else:
            return tree

# XGBoost模型类
class XGBoostModel:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3, lambda_reg=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lambda_reg = lambda_reg
        self.trees = []

    def fit(self, X, y):
        y_pred = np.zeros_like(y, dtype=np.float64)
        for i in range(self.n_estimators):
            g = gradient(y, y_pred)
            h = hessian(y, y_pred)

            tree = XGBoostTree(max_depth=self.max_depth, lambda_reg=self.lambda_reg)
            tree.fit(X, g, h)

            update = tree.predict(X)
            y_pred += self.learning_rate * update

            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

# 训练模型
model = XGBoostModel(n_estimators=50, learning_rate=0.1, max_depth=3, lambda_reg=1)
model.fit(X, y)

# 预测和可视化
y_pred = model.predict(X)

plt.figure(figsize=(10, 5))

# 可视化预测值与真实值的关系
plt.subplot(1, 2, 1)
plt.scatter(y, y_pred, color='blue', alpha=0.5)
plt.title('Predicted vs Actual Sale Prices')
plt.xlabel('Actual Sale Prices')
plt.ylabel('Predicted Sale Prices')

# 可视化残差
residuals = y - y_pred
plt.subplot(1, 2, 2)
plt.hist(residuals, bins=50, color='red', alpha=0.7)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

plt.show()