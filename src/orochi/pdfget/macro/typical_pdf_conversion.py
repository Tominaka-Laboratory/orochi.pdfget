import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

from orochi.pdfget.pdfget import PDFget, par as mpar, calculator as matrialpdf
from orochi.pdfget.macro import minimize_ripple


def typical_process_pixia_data(
    sample_file, pdf_par, out_path, rmin=None, rmax=None, background_file=None
):
    """Processes a 3-column Pixia output file (2theta, I, dummy_baseline) into G(r)."""
    os.makedirs(out_path, exist_ok=True)
    file_name = os.path.basename(sample_file).split(".")[0]
    data = read_a_file(sample_file)
    bg = read_a_file(background_file) if background_file is not None else []

    mat_cal = pixia_data(data, pdf_par, bg)

    savename = f"{out_path}/{file_name}_mpdf_data.txt"
    save_Gr(savename, mat_cal, rmin, rmax)

    basename = f"{out_path}/{file_name}"
    save_figures(basename, mat_cal, rmin, rmax)
    return mat_cal


def pixia_data(data, pdf_par, bg=None):
    """Runs PDF reduction using a 3-column array [2theta, I, dummy_baseline]."""
    mat_cal = matrialpdf(pdf_par)
    mat_cal.data["tt"] = data[:, 0]
    mat_cal.data["i"] = data[:, 1]

    if bg is not None and np.size(bg) != 0:
        mat_cal.background["tt"] = bg[:, 0]
        mat_cal.background["i"] = bg[:, 1]

    mat_cal["base_function_type"] = "data"
    mat_cal.data["base_function"] = data[:, 2]

    mat_cal = minimize_ripple.tune(mat_cal)
    mat_cal.run()
    return mat_cal


def typical_process_i_tt(
    sample_file, pdf_par, out_path, rmin=None, rmax=None, background_file=None
):
    """Processes a standard 2-column 1D diffraction file (2theta, I) into G(r)."""
    os.makedirs(out_path, exist_ok=True)
    file_name = os.path.basename(sample_file).split(".")[0]
    data = read_a_file(sample_file)
    bg = read_a_file(background_file) if background_file is not None else []

    mat_cal = general_data(data, pdf_par, bg)

    savename = f"{out_path}/{file_name}_mpdf_data.txt"
    save_Gr(savename, mat_cal, rmin, rmax)

    basename = f"{out_path}/{file_name}"
    save_figures(basename, mat_cal, rmin, rmax)
    return mat_cal


def general_data(data, pdf_par, bg=None):
    """Runs PDF reduction using a 2-column array [2theta, I]."""
    mat_cal = matrialpdf(pdf_par)
    mat_cal.data["tt"] = data[:, 0]
    mat_cal.data["i"] = data[:, 1]

    if bg is not None and np.size(bg) != 0:
        mat_cal.background["tt"] = bg[:, 0]
        mat_cal.background["i"] = bg[:, 1]

    mat_cal = minimize_ripple.tune(mat_cal)
    mat_cal.run()
    return mat_cal


def save_Gr(file_name, mat_cal, rmin, rmax):
    """Saves G(r) over [rmin, rmax] to a text file."""
    x = mat_cal.data["r"]
    y = mat_cal.data["Gr"]
    rmin_val = rmin if rmin is not None else float(x[0])
    rmax_val = rmax if rmax is not None else float(x[-1])
    use = np.where((x >= rmin_val) & (x <= rmax_val))
    data = np.column_stack([x[use], y[use]])
    np.savetxt(file_name, data, fmt="%.5e")


def save_figures(basename, mat_cal, rmin, rmax):
    """Saves diagnostic plots for G(r), F(Q), S(Q), I(Q), and form-factor fitting."""
    x = mat_cal.data["r"]
    y = mat_cal.data["Gr"]
    rmin_val = rmin if rmin is not None else float(x[0])
    rmax_val = rmax if rmax is not None else float(x[-1])
    use = np.where((x >= rmin_val) & (x <= rmax_val))
    y = y[use]
    x = x[use]

    fig = plt.figure()
    plt.plot(x, y, label="Gr")
    plt.legend()
    name = basename + "_Gr_mpdf"
    plt.savefig(name + ".png")
    save_pkl(fig, name)
    plt.close()

    fig = plt.figure()
    plt.plot(mat_cal.data["q_fq"], mat_cal.data["fq"], label="fq")
    plt.legend()
    name = basename + "_fq_mpdf"
    plt.savefig(name + ".png")
    save_pkl(fig, name)
    plt.close()

    fig = plt.figure()
    plt.plot(mat_cal.data["q"], mat_cal.data["sq"], label="sq")
    plt.legend()
    name = basename + "_sq_mpdf"
    plt.savefig(name + ".png")
    save_pkl(fig, name)
    plt.close()

    fig = plt.figure()
    plt.plot(mat_cal.data["q"], mat_cal.data["iq"], label="iq")
    plt.plot(
        mat_cal.data["q"],
        mat_cal.data["i_corrected"],
        ls=":",
        label="corrected",
    )
    plt.legend()
    name = basename + "_iq_mpdf"
    plt.savefig(name + ".png")
    save_pkl(fig, name)
    plt.close()

    fig = plt.figure()
    plt.plot(mat_cal.data["q"], mat_cal.data["f_sq_ave"], label="<f^2>")
    plt.plot(
        mat_cal.data["q"], mat_cal.data["f_ave_sq"], ls="--", label="<f>^2"
    )
    plt.plot(
        mat_cal.data["q"], mat_cal.data["compton"], ls=":", label="compton"
    )
    plt.plot(
        mat_cal.data["q"],
        mat_cal.data["base_function_normalized"],
        ls=":",
        label="base_function",
    )
    plt.plot(
        mat_cal.data["q"],
        mat_cal.data["i_normalized"],
        ls="-",
        label="normalized",
    )
    plt.legend()
    name = basename + "_ff_mpdf"
    plt.savefig(name + ".png")
    save_pkl(fig, name)
    plt.close()


def save_pkl(fig, file_name):
    with open(file_name + "_fig.pkl", "wb") as ofs:
        pickle.dump(fig, ofs)


def read_a_file(file):
    ext = os.path.splitext(file)[-1]
    if ext in [".csv", ".CSV"]:
        return np.genfromtxt(file, comments="#", delimiter=",")
    return np.genfromtxt(file, comments="#")
