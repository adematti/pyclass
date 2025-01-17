pyclass
-------

Another Python binding of the CMB Boltzmann code `CLASS`, v3, with access to velocity power spectra.

.. _`CLASS` : http://class-code.net

Dependencies
------------

- numpy
- cython

The CLASS code will be downloaded and compiled at installation.

Installation
------------

.. code:: bash

   python -m pip install git+https://github.com/adematti/pyclass

Mac OS
------
If you wish to use clang compiler (instead of gcc), you may encounter an error related to ``-fopenmp`` flag.
In this case, you can try to export:

.. code:: bash

   export CC=clang

Before installing **pyclass**. This will set clang OpenMP flags for compilation (see https://github.com/lesgourg/class_public/issues/405). Note that with Mac OS gcc can point to clang.

Examples
--------

See the tests of the code in ``pyclass/tests/`` for examples of using each of the main CLASS modules.

Extensions
----------
To implement a CLASS extension ``extension``:

- copy-paste ``base`` into e.g. ``extension``
- update the version and url in ``extension/_version.py``
- if necessary, make the changes to the Cython ``extension/binding.pyx`` and ``extension/cclassy.pxd`` files for the specific extension

That's it! Upon installation (``pip install --user .``), the extension is automatically compiled (together with ``base``), and can be imported as:

.. code:: python

   from pyclass.extension import *

Acknowledgments
----------------

This code heavily relies on classylss: http://classylss.readthedocs.io/.
