"""Equation for polarization correction.

References
----------
Fundamentals of Crystallography, 3rd ed., pp. 384-.
"""


import numpy as np

# q: amplitude of scattering vector (momentum transfer)
# par: MPDpar


def polarization_correction(q, args):
    tt = np.arcsin(q / (4.0 * np.pi / args["rambda"])) * 2.0

    if args["source"] in ["sr", "synchrotron"]:
        sqcostt = np.square(np.cos(tt))
        sqsintt = np.square(np.sin(tt))
        polarization = (1.0 + sqcostt + args["polar"] * sqsintt) / 2.0
        return polarization

    else:
        sqcostt = np.square(np.cos(tt))
        monochro = np.abs(
            np.cos(args["monochro_angle"]) ** args["monochro_mosaic"]
        ) * (1.0 - args["polar"])
        polarization = (1.0 + monochro * sqcostt) / (1.0 + monochro)
        return polarization
