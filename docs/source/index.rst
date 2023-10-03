Welcome to confounding-robust-inference's documentation!
========================================================

.. toctree::
   :hidden:

   installation

.. toctree::
   :hidden:

   examples/index

.. toctree::
   :maxdepth: 2
   :hidden:

   api_reference/index

.. toctree::
   :hidden:

   license

Introduction
------------

This package implements confounding robust estimators discussed in Ishikawa, et. al. (2023) [2]_ and its preliminary version [1]_.
These estimators robustifies the classical inverse probability weighting estimator, by considering the worst-case confounding and provide lower bound (or upper bound) of the policy value.

The supported estimators includes: inverse probability weighting (IPW) estimator, Hajek estimator, kernel conditional moment constraint (KCMC) estimator, and quantile balancing (QB) estimator.
For the exhausive list of supported estimators, see :doc:`api_reference/estimators`.

Referece
--------
.. [1] `"Kei Ishikawa, and Niao He. "Kernel Conditional Moment Constraints for Confounding Robust Inference." International Conference on Artificial Intelligence and Statistics. PMLR, 2023." <https://arxiv.org/abs/2302.13348>`_
.. [2] `"Kei Ishikawa, Naio He, and Takafumi Kanamori. "A Convex Framework for Confounding Robust Inference." arXiv preprint arXiv:2309.12450. 2023." <https://arxiv.org/abs/2309.12450>`_
