"""X-ray Compton scattering calculation.

References
----------
B. E. Warren, "X-ray Diffraction", Addison-Wesley (1969/1990).
T. Egami and S. J. L. Billinge, "Underneath the Bragg Peaks", Eq. 5.12.
"""


import numpy as np

cons_h = 6.62606957 * 10 ** (-34)  # Plank constant, [J s]
cons_c = 2.99792458 * 10**8  # speed of light in vacuum [m/s]
cons_ec = 1.602176565 * 10**-19  # elementary charge [C]
cons_avg = 6.022140857 * 10**23.0  # avogadro number [mol-1]
cons_me = 9.1093821 * 10 ** (-31)  # mass of electron [kg]


# Breit-Dirac recoil factor (Ec/E0)^a, a=2 flux, a=3 intensity
# Ec/E0=1/(1+E0/me/c^2*(1-cos)) , (E0-Ec)/E0Ec = (1-cos)/me/c^2


def compton_calculation(symbols, concentration, q, energy):
    from orochi.phase.submodules import atom_info as phase_utility

    symbols = np.array([symbols]).flatten()
    concentration = np.array([concentration]).flatten()

    concs = {}  # dictionary type {atom type: number}
    for n, atom in enumerate(symbols):
        if atom in concs.keys():
            concs[atom] += concentration[n]
        else:
            concs[atom] = concentration[n]

    electrons = phase_utility.atom_info_dicionary(
        symbols, key="electron"
    )  # dictionary type {atom type: number}
    comptons = phase_utility.atom_info_dicionary(
        symbols, key="compton"
    )  # dictionary type {atom type: function(q)}

    cf = np.zeros_like(q)
    total_electrons = 0.0

    for atom, conc in list(concs.items()):
        total_electrons += electrons[atom] * conc
        cf += comptons[atom](q) * conc

    csum = np.sum(list(concs.values()))
    total_electrons /= csum
    e2 = np.square(total_electrons)
    cf /= csum
    cf /= e2

    mec2 = cons_me * cons_c**2.0 / (1.60217733 * 10**-19.0)  # 0.510998910 MeV
    rambda = 1.0 / (energy / cons_c / cons_h * cons_ec / 10**10.0)
    theta = np.arcsin((q / (4.0 * np.pi) * rambda))
    ec_e0 = 1.0 / (1.0 + (energy / mec2) * (1.0 - np.cos(theta)))

    return cf, ec_e0
