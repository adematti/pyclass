pyclass
-------

|

Another Python binding of the CMB Boltzmann code `CLASS`, v3, with access to velocity power spectra.

.. _`CLASS` : http://class-code.net

Dependencies
------------

- numpy
- cython

The CLASS code will be downloaded at installation.
The CLASS version can be accessed through ``pyclass.class_version``.

Installation
------------

.. code:: bash

   pip install git+https://github.com/adematti/pyclass

Examples
--------

See the tests of the code in ``pyclass/tests/`` for examples of using each of the main CLASS modules.

Acknowledgments
----------------

This code heavily relies on classylss: http://classylss.readthedocs.io/.
