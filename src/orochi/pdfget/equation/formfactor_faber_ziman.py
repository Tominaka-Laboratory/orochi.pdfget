"""X-ray scattering structure factor and form factor in Faber-Ziman formalism.

References
----------
T. E. Faber and J. M. Ziman, Philos. Mag. 11, 153-173 (1965).
"""


import numpy as np

# Faber T Ziman J Philosophical Magazine 1965 vol: 11 (109) pp: 153-173
# S(Q)={I(Q)-(<f^2>-<f>^2)}/<f>^2={I(Q)-<f^2>}/<f>^2+1
# <f^2> #Laue diffuse scattering = <f^2>-<f>^2

# return <f2>, <f>^2, (normalized by electrons^2 (average number of electrons)


def form_factor_calculation(symbols, concentration, q, energy):
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
    f0s = phase_utility.atom_info_dicionary(
        symbols, key="f0"
    )  # dictionary type {atom type: function(q)}
    f1f2s = phase_utility.atom_info_dicionary(
        symbols, key="f1f2"
    )  # dictionary type {atom type: function(energy)}

    average_square = np.zeros_like(q) + 0.0j
    square_average = np.zeros_like(q) + 0.0j
    total_electrons = 0.0

    for atom, conc in list(concs.items()):
        total_electrons += electrons[atom] * conc

        f0 = f0s[atom](q)
        f1, f2 = f1f2s[atom](energy)
        sf = f0 + f1 + 1.0j * f2

        average_square += sf * conc
        square_average += np.square(sf) * conc

    csum = np.sum(list(concs.values()))
    total_electrons /= csum
    e2 = np.square(total_electrons)

    fas = np.abs(np.square(average_square / csum)) / e2
    fsa = np.abs(square_average / csum) / e2

    return fsa, fas
