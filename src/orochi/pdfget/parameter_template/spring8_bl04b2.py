import numpy as np


def use_2016b(mpar):
    mpar["rambda"] = 0.20231
    mpar["qmax"] = 23.66
    mpar["qmin"] = 0.267
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
