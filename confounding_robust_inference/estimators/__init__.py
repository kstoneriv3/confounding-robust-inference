from confounding_robust_inference.estimators.base import BaseEstimator, BaseKCMCEstimator
from confounding_robust_inference.estimators.baselines import (
    HajekEstimator,
    IPWEstimator,
    QBEstimator,
    ZSBEstimator,
)
from confounding_robust_inference.estimators.kcmc import (
    DualKCMCEstimator,
    DualKCMCPolicyLearner,
    GPKCMCEstimator,
    KCMCEstimator,
)
from confounding_robust_inference.estimators.ncmc import DualNCMCEstimator
from confounding_robust_inference.utils.docs import _populate_docstrings

__all__ = [
    "BaseEstimator",
    "BaseKCMCEstimator",
    "DualKCMCEstimator",
    "DualKCMCPolicyLearner",
    "DualNCMCEstimator",
    "GPKCMCEstimator",
    "HajekEstimator",
    "IPWEstimator",
    "KCMCEstimator",
    "QBEstimator",
    "ZSBEstimator",
]

for _cls_name in __all__:
    _cls = vars()[_cls_name]
    _populate_docstrings(_cls)
