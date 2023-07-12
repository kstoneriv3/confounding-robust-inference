Installation
============

To use and develop the confounding-robust-inference, download the repository and install it as an editable package:

.. code-block:: bash

    $ git clone git@github.com:kstoneriv3/confounding-robust-inference.git
    $ cd ./confounding-robust-inference
    $ pip install -e .

.. note::
   A commercial solver called used `MOSEK <https://www.mosek.com/downloads/>`_ can be used as the backend of CVXPY.
   The developer of confounding-robust-inference found MOSEK tends to produce more high-quality solutions for non-linear convex optimization problems associated to the conditional f-sensitivity models.

   For people in academia, MOSEK offers a `free academic license for trial <https://www.mosek.com/products/academic-licenses/>`_.

   However, it is completely optional and a reasonable solutions can usually be obtained without installing it.
