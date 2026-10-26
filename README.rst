.. image:: images/orochi_icon.png
   :align: center
   :width: 80px
   :target: https://Tominaka-Laboratory.github.io/orochi.pdfget
   :alt: Orochi Logo

=============
orochi.pdfget
=============

.. |title| replace:: orochi.pdfget
.. _title: https://Tominaka-Laboratory.github.io/orochi.pdfget

.. |Icon| image:: images/orochi_icon.png
   :target: https://Tominaka-Laboratory.github.io/orochi.pdfget
   :height: 100px

|PyPi| |Forge| |PythonVersion| |PR|

|CI| |Codecov| |Black| |Tracking|

.. |Black| image:: https://img.shields.io/badge/code_style-black-black
        :target: https://github.com/psf/black

.. |CI| image:: https://github.com/Tominaka-Laboratory/orochi.pdfget/actions/workflows/tests.yml/badge.svg
        :target: https://github.com/Tominaka-Laboratory/orochi.pdfget/actions/workflows/tests.yml

.. |Codecov| image:: https://codecov.io/gh/Tominaka-Laboratory/orochi.pdfget/branch/main/graph/badge.svg
        :target: https://codecov.io/gh/Tominaka-Laboratory/orochi.pdfget

.. |Forge| image:: https://img.shields.io/conda/vn/conda-forge/orochi.pdfget
        :target: https://anaconda.org/conda-forge/orochi.pdfget

.. |PR| image:: https://img.shields.io/badge/PR-Welcome-29ab47ff

.. |PyPi| image:: https://img.shields.io/pypi/v/orochi.pdfget
        :target: https://pypi.org/project/orochi.pdfget/

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/orochi.pdfget
        :target: https://pypi.org/project/orochi.pdfget/

.. |Tracking| image:: https://img.shields.io/badge/issue_tracking-github-blue
        :target: https://github.com/Tominaka-Laboratory/orochi.pdfget/issues

Orochi PDFget: 1D Total Scattering Reduction & Faber–Ziman Normalization
========================================================================

The ``orochi.pdfget`` package provides rigorous 1D total scattering reduction, transforming raw experimental scattering profiles *I(2θ)* or *I(Q)* into the total structure factor *S(Q)*, reduced structure factor *F(Q)* and atomic pair distribution function *G(r)* using the Faber–Ziman formalism.

Key Features
------------
* **Background & Solvent Halo Subtraction**: Performs scaling and subtraction of empty container, air, or liquid solvent scattering.
* **Compton Recoil & Fluorescence Correction**: Subtracts incoherent Compton inelastic scattering and flat fluorescence baselines.
* **High-Q Form-Factor Normalization**: Normalizes experimental intensity against theoretical atomic scattering ``<|f(Q)|^2> + I_Compton(Q)`` over a user-defined high-Q polynomial window [Q_fit,min, Q_fit,max].
* **Fourier Transformation via DST-I**: Implements the Type-I Discrete Sine Transform (DST-I) with optional Lorch modification functions to minimize termination ripples without introducing phase shifts.

Quickstart: 1D Reduction to G(r)
================================

.. code-block:: python

    from orochi.pdfget import PDFget
    import matplotlib.pyplot as plt

    # 1. Initialize PDFget
    pdfget = PDFget()
    pdfget.read_file("ceo2_1d_profile.txt", target="sample")

    # 2. Configure sample chemistry and reduction interval
    pdfget["composition"] = "CeO2"
    pdfget["rambda"] = 0.18033           # Incident wavelength in Angstrom
    pdfget["qmin"] = 1.0                 # Integration bounds
    pdfget["qmax"] = 25.0
    pdfget["fitmin"] = 15.0              # Normalization window
    pdfget["fitmax"] = 25.0
    pdfget["rmin"] = 0.0
    pdfget["rmax"] = 30.0

    # 3. Execute 1D reduction pipeline
    pdfget.run()

    # 4. Access reduced structure factor F(Q) and G(r)
    q_grid = pdfget.q
    fq_data = pdfget.fq

    r_grid = pdfget.r
    gr_data = pdfget.gr

    plt.plot(r_grid, gr_data)
    plt.xlabel("r (Å)")
    plt.ylabel("G(r) (Å⁻²)")
    plt.show()

Citation
--------

If you use orochi.pdfget in a scientific publication, please cite this package as:

    S. Tominaka, *Orochi: A Physics-Grounded Python Package for Pair Distribution Function Analysis of Crystalline and Low-Dimensional Materials*, 2026.
    https://github.com/Tominaka-Laboratory/orochi.pdfget

Installation
------------

The preferred method is to install using ``pip``:

.. code-block:: bash

    pip install orochi.pdfget

To confirm that the installation was successful, type:

.. code-block:: bash

    python -c "import orochi.pdfget; print('Orochi PDFget imported successfully')"

If you prefer to install from source:

.. code-block:: bash

    git clone https://github.com/Tominaka-Laboratory/orochi.pdfget.git
    cd orochi.pdfget
    pip install -e .

Getting Started
---------------

You may consult our `online documentation <https://Tominaka-Laboratory.github.io/orochi.pdfget>`_ for tutorials and API references.

Support and Contribute
----------------------

The orochi.pdfget project welcomes your expertise and enthusiasm!

If you see a bug or want to request a feature, please `report it as an issue <https://github.com/Tominaka-Laboratory/orochi.pdfget/issues>`_ and/or `submit a fix as a PR <https://github.com/Tominaka-Laboratory/orochi.pdfget/pulls>`_. 

Feel free to fork the project and contribute. To install orochi.pdfget
in a development mode, with its sources being directly used by Python
rather than copied to a package directory, use the following in the root
directory ::

        pip install -e .

To ensure code quality and to prevent accidental commits into the default branch, please set up the use of our pre-commit
hooks.

Improvements and fixes are always appreciated.

Before contributing, please read our `Code of Conduct <https://github.com/Tominaka-Laboratory/orochi.pdfget/blob/main/CODE_OF_CONDUCT.rst>`_.



License
-------

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**. See the `LICENSE.rst` file for details.

This copyleft license guarantees that Orochi remains freely accessible, open, and transparent to the entire scientific community, and that all derivative works must likewise remain open source.

Citation
--------

If you use this package in academic research, please cite:

.. code-block:: bibtex

    @article{tominaka2026orochi,
      title   = {Orochi: A Physics-Grounded Python Package for Pair Distribution Function Analysis of Crystalline and Low-Dimensional Materials},
      author  = {Tominaka, Satoshi},
      year    = {2026},
      url     = {https://github.com/Tominaka-Laboratory/orochi}
    }

Contact
-------

For more information on orochi.pdfget please visit the project `web-page <https://Tominaka-Laboratory.github.io/>`_ or email Dr. Satoshi Tominaka at tominaka.satoshi@nims.go.jp.
