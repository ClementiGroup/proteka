proteka
==========


.. start-intro
Library for comparing and benchmarking protein models. In particular
it contains an implementation for computing the fraction of native contacts
using as a reference a CG pdb.


.. end-intro

Installation
------------
.. start-install
The dependencies are defined in `requirements.txt` so it can be install with pip::

    pip install git+https://github.com/ClementiGroup/proteka.git

or by cloning the repository and installing localy::

    git clone https://github.com/ClementiGroup/proteka.git
    pip install -e .


.. end-install


.. start-doc

Documentation
-------------

Documentation is available `here <https://clementigroup.github.io/proteka/>`_ and here are some references on how to work with it.

Dependencies
~~~~~~~~~~~~

.. code:: bash

    pip install sphinx sphinx_rtd_theme sphinx-autodoc-typehints


How to build
~~~~~~~~~~~~

.. code:: bash

    cd docs
    sphinx-build -b html source build

How to update the online documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This udapte should be done after any update of the `main` branch so that the
documentation is synchronised with the main version of the repository.

.. code:: bash

    git checkout gh-pages
    git rebase main
    cd docs
    sphinx-build -b html source ./
    git commit -a
    git push

.. end-doc
