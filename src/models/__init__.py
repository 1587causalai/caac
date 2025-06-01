"""
共享潜在柯西向量的 One-vs-Rest (OvR) 多分类器模型

提供以下组件：
1. FeatureNetwork - 特征提取网络
2. AbductionNetwork - 推断网络
3. LinearTransformationLayer - 线性变换层
4. OvRProbabilityLayer - OvR概率计算层
5. UnifiedClassificationNetwork - 统一分类网络
6. CAACOvRModel - 模型包装类
"""

from .feature_network import FeatureNetwork
from .abduction_network import AbductionNetwork
from .linear_transformation import LinearTransformationLayer
from .ovr_probability import OvRProbabilityLayer
from .unified_network import UnifiedClassificationNetwork
from .caac_ovr_model import CAACOvRModel

__all__ = [
    'FeatureNetwork',
    'AbductionNetwork',
    'LinearTransformationLayer',
    'OvRProbabilityLayer',
    'UnifiedClassificationNetwork',
    'CAACOvRModel',
]

# Models module
