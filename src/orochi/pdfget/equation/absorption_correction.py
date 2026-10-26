"""Equation for sample absorption correction.

References
----------
Equation for capillary samples in I. Nakai and F. Izumi (Eds.),
"Powder X-ray Diffraction: Analysis by the Rietveld Method".
"""


import numpy as np

# q: amplitude of scattering vector (momentum transfer)
# par: MPDpar


def absorption_correction(q, rambda, mur):
    x = q / 4.0 / np.pi * rambda
    absorption = 1.0 / (
        np.exp(
            -1.0 * (1.7133 - 0.0368 * x) * mur
            + (0.0927 + 0.375 * x) * np.square(mur)
        )
    )
    return absorption
