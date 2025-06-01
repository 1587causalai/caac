import sys
import os
# 添加项目根目录到Python路径，以便导入src模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.models.caac_ovr_model import CrammerSingerMLPModel

print('Testing CrammerSingerMLPModel...')

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建并训练模型
model = CrammerSingerMLPModel(
    input_dim=X_train.shape[1],
    n_classes=len(np.unique(y_train)),
    representation_dim=32,
    epochs=50,
    lr=0.01
)

print('Training...')
model.fit(X_train, y_train, verbose=1)

print('Predicting...')
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f'Test Accuracy: {accuracy:.4f}')

print('Testing predict_proba...')
probs = model.predict_proba(X_test)
print(f'Probabilities shape: {probs.shape}')
print(f'Probabilities sum (first sample): {probs[0].sum():.4f}')

print('CrammerSingerMLPModel test completed successfully!') 