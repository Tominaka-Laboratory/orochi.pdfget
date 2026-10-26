import numpy as np


def use_2018a(mpar):
    mpar["rambda"] = 0.10760
    mpar["qmax"] = 35
    mpar["qmin"] = 1.64
    mpar["qstep"] = 0.002
    mpar["fitmax"] = 35
    mpar["fitmin"] = 22
    mpar["mur"] = 0.0
    mpar["polar"] = 1.0
    mpar["lorch"] = 5.0
    mpar["number_density"] = 0.069206
    mpar["qfilter"] = 0.0
    mpar["rfilter"] = 0.0
    mpar["base_function_type"] = "data"
    return mpar
