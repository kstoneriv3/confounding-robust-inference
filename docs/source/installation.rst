Installation
============

To use and develop the confounding-robust-inference, download the repository and install it as an editable package:

.. code-block:: bash

    $ git clone git@github.com:kstoneriv3/confounding-robust-inference.git
    $ cd ./confounding-robust-inference
    $ pip install -e .

.. note::
   A commercial solver called used `MOSEK <https://www.mosek.com/downloads/>`_ can be used as the backend of CVXPY, to obtain stable solutions.
   The developer found MOSEK to be more stable when solving non-linear convex optimization problem associated to the conditional f-sensitivity models.
   For people in academia, there is a `free academic license for trial <https://www.mosek.com/products/academic-licenses/>`_.
   However, it is completely optional and a reasonable solutions can usually be obtained without installing it.
