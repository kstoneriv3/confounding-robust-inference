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

Intro
-----

This package implements confounding robust estimators discussed in [1]_.
Such estimators robustifies the classical inverse probability weighting estimator, by considering the worst-case confounding and provide lower bound (or upper bound) of the policy value.

The supported estimators includes: inverse probability weighting (IPW) estimator, Hajek estimator, kernel conditional moment constraint (KCMC) estimator.
For the exhausive list of supported estimators, see :doc:`api_reference/estimators`.

Referece
--------
.. [1] `"Ishikawa, Kei, and Niao He. "Kernel Conditional Moment Constraints for Confounding Robust Inference." International Conference on Artificial Intelligence and Statistics. PMLR, 2023." <https://arxiv.org/abs/2302.13348>`_
