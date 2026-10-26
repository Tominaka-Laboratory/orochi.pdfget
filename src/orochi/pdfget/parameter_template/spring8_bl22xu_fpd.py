import numpy as np


def use_2023a(mpar):
    mpar["rambda"] = 0.181050
    mpar["qmax"] = 25
    mpar["qmin"] = 1.3
    mpar["qstep"] = 0.002
    mpar["fitmax"] = 25
    mpar["fitmin"] = 6
    mpar["mur"] = 0.0
    mpar["polar"] = 1.0
    mpar["lorch"] = 5.0
    mpar["number_density"] = 0.069206
    mpar["qfilter"] = 0.0
    mpar["rfilter"] = 0.0
    mpar["baseq"] = 5
    mpar["basetype"] = "flat"
    return mpar


def use_2024a(mpar):
    mpar["rambda"] = 0.18123949345740054
    mpar["qmax"] = 25
    mpar["qmin"] = 1.3
    mpar["qstep"] = 0.002
    mpar["fitmax"] = 25
    mpar["fitmin"] = 8
    mpar["mur"] = 0.0
    mpar["polar"] = 1.0
    mpar["lorch"] = 5.0
    mpar["number_density"] = 0.069206
    mpar["qfilter"] = 0.0
    mpar["rfilter"] = 0.0
    mpar["base_function_type"] = "data"
    return mpar
