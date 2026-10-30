import math
import os
import re
import time
import json
import numpy as np
from numpy import pi
from scipy import interpolate as intp
from scipy import signal, ndimage
from scipy.fftpack import dst, idst

from orochi.base.core import OrochiBase
from .params import default
from .calculator import form_factor_fitting as fit


class PDFget(OrochiBase):
    """Total scattering 1D data reduction module for extracting S(Q), F(Q), and G(r).

    Implements the physics-grounded Faber-Ziman data reduction workflow:
    1. Scattering angle (2theta) to momentum transfer (Q) conversion with anti-aliasing.
    2. Container background subtraction, capillary absorption, and polarization corrections.
    3. Relativistic Compton scattering and detector/fluorescence baseline isolation via
       high-Q atomic form factor normalization.
    4. Automated low-r ripple minimization over [fitmin, fitmax] using Optuna TPE.
    5. Revised Lorch modification and Type-I Discrete Sine Transform (DST-I) to G(r).
    """

    def __init__(self):
        self._par = default.par
        self._par_key_info = default.par_key_info
        super().__init__()

        self._name = "orochi.pdfget"
        self._id = "0"

        self.ttraw = 0.0
        self.iraw = 0.0
        self.bg = 0.0

        self.data = {
            "tt": np.empty(0),
            "i": np.empty(0),
            "base_function": np.empty(0),
        }
        self.background = {
            "tt": np.empty(0),
            "i": np.empty(0),
        }

        self.set_source()
        rambda_val = self["rambda"]
        energy_val = self["energy"]
        if (rambda_val is None or np.isnan(rambda_val)) and (
            energy_val is not None and not np.isnan(energy_val)
        ):
            self.set_rambda_from_energy()
        elif (rambda_val is not None and not np.isnan(rambda_val)) and (
            energy_val is None or np.isnan(energy_val)
        ):
            self.set_energy_from_rambda()

        self.set_fitmin_max()
        self._set_equations()
        if "number_density" in self.par:
            self.par["ndensity"] = self.par["number_density"]

    def __getitem__(self, key):
        base_key = key[0] if isinstance(key, (list, tuple)) else key
        if base_key == "ndensity":
            if isinstance(key, (list, tuple)):
                key = ("number_density", *key[1:])
            else:
                key = "number_density"
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        base_key = key[0] if isinstance(key, (list, tuple)) else key
        if base_key == "ndensity":
            if isinstance(key, (list, tuple)):
                key = ("number_density", *key[1:])
            else:
                key = "number_density"
            base_key = "number_density"

        res = super().__setitem__(key, value)

        if base_key == "number_density" and "ndensity" in self.par:
            self.par["ndensity"] = self.par["number_density"]

        if base_key == "rambda" and self["rambda"] is not None:
            self.set_energy_from_rambda()
        elif base_key == "energy" and self["energy"] is not None:
            self.set_rambda_from_energy()
        elif base_key == "source":
            self.set_source()
        elif base_key == "composition" and isinstance(value, str):
            self._parse_composition(value)
        elif base_key == "qmax" and self["qmax"] is not None:
            if self["fitmax"] is None or np.isnan(self["fitmax"]):
                self["fitmax"] = float(self["qmax"])
            if self["fitmin"] is None or np.isnan(self["fitmin"]):
                self["fitmin"] = float(self["qmax"]) / 2.0
        return res

    def __call__(self, force: bool = False):
        """Execute the full 1D total scattering to G(r) data reduction pipeline.

        Performs:
        1. 2theta to Q conversion and anti-aliasing interpolation.
        2. Background subtraction and sample absorption/polarization correction.
        3. Form factor and Compton scattering normalization over [fitmin, fitmax].
        4. Baseline extraction and S(Q) -> F(Q) conversion.
        5. Revised Lorch windowing and Discrete Sine Transform to G(r).

        Parameters
        ----------
        force : bool, optional
            If True, resets internal state to 'INIT' to force full re-evaluation, by default False.
        """
        if force:
            self._state = "INIT"
        super().__call__(force=force)

        self._logger.info("PDFget call start...")
        self._set_equations()

        # Check if linked to pixia source
        if self.data["tt"].size == 0 and getattr(self, "_pixia_source", None) is not None:
            self._sync_from_pixia()

        if self.data["tt"].size == 0 and self["sample_file"] is not None:
            self.read_file(self["sample_file"], target="sample")
        if self.background["tt"].size == 0 and self["background_file"] is not None:
            self.read_file(self["background_file"], target="background")

        if self.data["tt"].size == 0:
            raise RuntimeError("No input 1D scattering data ('tt', 'i') loaded in PDFget.")

        self.data = self.get_iq(self.data)

        if self.background["tt"].size > 0:
            self.background = self.get_iq(self.background)

        self.data = self.get_i_bg_sub(self.data, self.background, self["bg_scale"])
        self.data = self.get_i_corrected(self.data)

        self.data = self.get_form_factor_compton(self.data)
        self.data = self.get_base_function(self.data)

        self.data = self.get_i_normalized(self.data)
        self.data = self.get_fq(self.data)
        self.data = self.get_Gr(self.data)
        self._state = "DONE"
        return self

    def run(self, force: bool = False):
        """Explicit execution entry point for user scripts and interactive environments.

        Parameters
        ----------
        force : bool, optional
            If True, forces full recalculation by bypassing cached state, by default False.
        """
        return self.__call__(force=force)

    @property
    def q(self):
        """Scattering vector grid Q for measured structure factors S(Q) and F(Q)."""
        if self.data and "q" in self.data:
            return self.data["q"]
        return self.data.get("q_fq") if self.data else None

    @property
    def q_fq(self):
        """Evenly spaced momentum transfer grid Q for Discrete Sine Transform (0 to qmax)."""
        return self.data.get("q_fq") if self.data else None

    @property
    def fq(self):
        """Measured reduced structure factor F(Q) on the experimental Q grid (no extrapolation)."""
        return self.data.get("fq") if self.data else None

    @property
    def fq_grid(self):
        """Extrapolated reduced structure factor F(Q) on the Fourier transform grid q_fq."""
        if self.data and "fq_grid" in self.data:
            return self.data["fq_grid"]
        return self.data.get("fq") if self.data else None

    @property
    def sq(self):
        """Total structure factor S(Q) on the measured Q grid (no extrapolation)."""
        return self.data.get("sq") if self.data else None

    @property
    def r(self):
        """Real-space distance grid r for G(r)."""
        return self.data.get("r") if self.data else None

    @property
    def Gr(self):
        """Reduced pair distribution function G(r) = 4 * pi * r * (rho(r) - rho0)."""
        return self.data.get("Gr") if self.data else None

    @property
    def gr(self):
        """Atomic pair distribution function g(r) = rho(r) / rho0 (converges to 1 as r -> inf)."""
        return self.data.get("gr") if self.data else None

    @property
    def tr(self):
        """Total radial distribution function T(r) = 4 * pi * r * rho(r) = G(r) + 4 * pi * r * rho0."""
        return self.data.get("tr") if self.data else None

    @property
    def Tr(self):
        """Alias for tr: Total radial distribution function T(r)."""
        return self.tr

    @property
    def tt(self):
        """Input scattering angle 2theta (degrees)."""
        return self.data.get("tt") if self.data else None

    @property
    def i(self):
        """Input 1D scattering intensity."""
        return self.data.get("i") if self.data else None

    @property
    def tr(self):
        """Radial distribution function T(r)."""
        return self.data.get("tr") if self.data else None

    def _parse_composition(self, formula_str):
        """Parses a chemical formula string (e.g., 'CeO2', 'Ru1 O2') into symbol and occ arrays."""
        tokens = re.findall(r"([A-Z][a-z]?(?:\d*[+-])?)\s*([0-9.]*)", formula_str.strip())
        if not tokens:
            return
        symbols = []
        occs = []
        for sym, count_str in tokens:
            if not sym:
                continue
            symbols.append(sym)
            occs.append(float(count_str) if count_str != "" else 1.0)
        if symbols:
            self.par["symbol"]["value"] = np.array(symbols, dtype=object)
            self.par["occ"]["value"] = np.array(occs, dtype=object)
            self.par["occ"]["min"] = np.zeros(len(occs), dtype=object)
            self.par["occ"]["max"] = np.array(occs, dtype=object) * 2.0
            self.par["occ"]["fit"] = np.zeros(len(occs), dtype=object)
            self.par["occ"]["error"] = np.zeros(len(occs), dtype=object)

    def set_phase(self, phase):
        """Synchronize chemical composition and number density from a Phase object.

        Parameters
        ----------
        phase : orochi.phase.Phase
            Crystallographic phase model containing atomic site symbols, occupancies,
            and unit cell volume.
        """
        comp = phase.get_model_composition()
        if comp:
            symbols = list(comp.keys())
            occs = [float(v) for v in comp.values()]
            self.par["symbol"]["value"] = np.array(symbols, dtype=object)
            self.par["occ"]["value"] = np.array(occs, dtype=object)
            formula_str = "".join(
                f"{s}{int(o) if o == int(o) else o}" for s, o in zip(symbols, occs)
            )
            self.par["composition"]["value"] = formula_str
        if hasattr(phase, "model") and "xyz" in phase.model and hasattr(phase, "volume"):
            vol = float(phase.volume)
            if vol > 0:
                n_atoms = float(np.sum(phase.model["occ"]))
                self["number_density"] = n_atoms / vol

    def sync_from_pixia(self, pixia_instance=None, auto_run: bool = False, copy_params_only: bool = False):
        """Synchronizes radiation parameters (rambda, energy) and 1D profile data
        (2theta, intensity, synthetic baseline) from a Pixia instance.

        Parameters
        ----------
        pixia_instance : orochi.pixia.Pixia, optional
            Pixia instance. If None, uses previously linked ``self._pixia_source``.
        auto_run : bool, default False
            If True and Pixia 1D data is not yet calculated, executes Pixia integration.
        copy_params_only : bool, default False
            If True, only synchronizes parameters (rambda, energy) without copying 1D data.

        Returns
        -------
        self : PDFget
        """
        if pixia_instance is not None:
            self._pixia_source = pixia_instance
        else:
            pixia_instance = getattr(self, "_pixia_source", None)

        if pixia_instance is None:
            return self

        # 1. Synchronize radiation parameters (rambda, energy)
        if hasattr(pixia_instance, "par"):
            if "rambda" in pixia_instance.par and pixia_instance["rambda"] is not None:
                try:
                    self["rambda"] = float(np.ravel(pixia_instance["rambda"])[0])
                except Exception:
                    pass
            elif "energy" in pixia_instance.par and pixia_instance["energy"] is not None:
                try:
                    self["energy"] = float(np.ravel(pixia_instance["energy"])[0])
                except Exception:
                    pass

        if copy_params_only:
            return self

        # 2. Run Pixia integration if requested and needed
        if auto_run and (getattr(pixia_instance, "tt1d", None) is None or getattr(pixia_instance, "i1d", None) is None):
            try:
                pixia_instance()
            except Exception:
                pass

        # 3. Synchronize 1D profile arrays
        tt1d = getattr(pixia_instance, "tt1d", None)
        i1d = getattr(pixia_instance, "i1d", None)
        if tt1d is not None and i1d is not None and len(tt1d) > 0 and len(i1d) > 0:
            if getattr(pixia_instance, "d1d", None) is None:
                try:
                    pixia_instance.get_background_curve()
                except Exception:
                    pass

            self.data["tt"] = np.asarray(tt1d, dtype=float)
            self.data["i"] = np.asarray(i1d, dtype=float)
            d1d = getattr(pixia_instance, "d1d", None)
            if d1d is not None and len(d1d) == len(self.data["tt"]):
                self.data["base_function"] = np.asarray(d1d, dtype=float)
                self["base_function_type"] = "data"

        return self

    def _sync_from_pixia(self):
        """Backward-compatible private alias for sync_from_pixia."""
        return self.sync_from_pixia()

    def from_pixia(self, pixia_instance, auto_sync: bool = False):
        """Import or link 1D profile and synthetic baseline directly from a Pixia instance.

        Parameters
        ----------
        pixia_instance : orochi.pixia.Pixia
            Integrated 2D-to-1D Pixia object containing ``tt1d``, ``i1d``, and ``d1d``.
        auto_sync : bool, optional
            If True and pixia has not been run, runs pixia immediately. Default is False.
        """
        return self.sync_from_pixia(pixia_instance, auto_run=auto_sync)

    def read_file(self, file_path, target="sample"):
        """Read a 1D diffraction data file.

        Parameters
        ----------
        file_path : str
            Path to 2-column [2theta, I] or 3-column [2theta, I, baseline] ASCII file.
        target : str, optional
            Target data slot: "sample" or "background". Default is "sample".

        Raises
        ------
        FileNotFoundError
            If file_path does not exist.
        ValueError
            If file has fewer than 2 columns.
        """
        if file_path is None or not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == ".csv":
            raw = np.genfromtxt(file_path, comments="#", delimiter=",")
        else:
            raw = np.genfromtxt(file_path, comments="#")

        if raw.ndim == 1 or raw.shape[1] < 2:
            raise ValueError(f"Expected at least 2 columns (2theta, intensity) in {file_path}")

        if target == "sample":
            self["sample_file"] = file_path
            self.data["tt"] = raw[:, 0]
            self.data["i"] = raw[:, 1]
            if raw.shape[1] >= 3:
                self.data["base_function"] = raw[:, 2]
                self["base_function_type"] = "data"
            elif self["base_function_type"] == "data" and self.data["base_function"].size == 0:
                self["base_function_type"] = "flat"
        elif target == "background":
            self["background_file"] = file_path
            self.background["tt"] = raw[:, 0]
            self.background["i"] = raw[:, 1]

    def to_json(self, file_path=None):
        """Export computed reduction curves and module parameters to a JSON dictionary/file.

        Parameters
        ----------
        file_path : str, optional
            If specified, saves the JSON to this path.

        Returns
        -------
        dict
            Dictionary containing computed data arrays ('Gr', 'r', 'fq', 'q', 'sq', etc.)
            and a nested 'params' dictionary containing all module parameters.
        """
        json_data = {}
        for attr, key in [
            ("r", "r"),
            ("Gr", "Gr"),
            ("gr", "gr"),
            ("tr", "tr"),
            ("q", "q"),
            ("sq", "sq"),
            ("fq", "fq"),
            ("q_fq", "q_fq"),
            ("fq_grid", "fq_grid"),
            ("tt", "tt"),
            ("i", "i"),
            ("base_function", "base_function"),
        ]:
            if attr in self.data and isinstance(self.data[attr], np.ndarray) and self.data[attr].size > 0:
                json_data[key] = self.data[attr].tolist()

        from orochi.base.core import NumpyEncoder
        params = {}
        for k in self.par:
            try:
                val = self[k]
                if isinstance(val, (np.bool_, bool)):
                    params[k] = bool(val)
                elif isinstance(val, np.ndarray):
                    params[k] = val.tolist()
                elif isinstance(val, (np.floating, float)):
                    params[k] = None if np.isnan(val) else float(val)
                elif isinstance(val, (np.integer, int)):
                    params[k] = int(val)
                elif isinstance(val, str) or val is None:
                    params[k] = val
                else:
                    params[k] = val
            except Exception:
                pass
        json_data["params"] = params

        if file_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            self._logger.info(f"Saved JSON data to {file_path}")

        return json_data

    def from_json(self, source):
        """Load module parameters and reduction data from a JSON dictionary or file.

        Parameters
        ----------
        source : str or dict
            File path or pre-loaded dictionary containing 'params' and data keys.
        """
        if isinstance(source, str):
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(source, dict):
            data = source
        else:
            raise TypeError("source must be a file path string or a dict")

        if "params" in data and isinstance(data["params"], dict):
            self.update(data["params"])

        key_map = {
            "r": "r",
            "Gr": "Gr",
            "gr": "gr",
            "tr": "tr",
            "q": "q",
            "sq": "sq",
            "fq": "fq",
            "q_fq": "q_fq",
            "fq_grid": "fq_grid",
            "tt": "tt",
            "i": "i",
            "base_function": "base_function",
        }
        for json_k, data_k in key_map.items():
            if json_k in data and data[json_k] is not None:
                self.data[data_k] = np.array(data[json_k])

        # Backward compatibility for old JSON where "q" held q_fq and "fq" held fq_grid
        if "q_fq" not in self.data and "q" in self.data:
            self.data["q_fq"] = self.data["q"]
        if "fq_grid" not in self.data and "fq" in self.data:
            self.data["fq_grid"] = self.data["fq"]

    def save_json(self, file_path):
        """Alias for to_json(file_path)."""
        return self.to_json(file_path)

    def load_json(self, file_path):
        """Alias for from_json(file_path)."""
        return self.from_json(file_path)

    def save_file(self, file_path):
        """Save reduced pair distribution function G(r) to a text file (.gr) or JSON (.json).

        Parameters
        ----------
        file_path : str
            Target file path for output file. If ends with .json, exports JSON format.

        Raises
        ------
        RuntimeError
            If G(r) has not been computed yet.
        """
        if file_path.endswith(".json"):
            return self.to_json(file_path)

        if "r" not in self.data or "Gr" not in self.data:
            raise RuntimeError("No G(r) data available to save. Run PDFget first.")
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        r = self.data["r"]
        gr = self.data["Gr"]
        rmin = self["rmin"] if self["rmin"] is not None else 0.0
        rmax = self["rmax"] if self["rmax"] is not None else float(r[-1])
        mask = (r >= rmin) & (r <= rmax)
        out_data = np.column_stack([r[mask], gr[mask]])
        header = (
            f"Qmin = {float(self['qmin']):.3f}\n"
            f"Qmax = {float(self['qmax']):.3f}\n"
            f"fitmin = {float(self['fitmin']):.3f}\n"
            f"fitmax = {float(self['fitmax']):.3f}\n"
            f"rmin = {float(rmin):.3f}\n"
            f"rmax = {float(rmax):.3f}\n\n"
            f"r(A) G(r)(A^-2)"
        )
        np.savetxt(file_path, out_data, fmt="%.5e", header=header)
        self._logger.info(f"Saved G(r) to {file_path}")

    def set_fitmin_max(self):
        """Initialize default high-Q fitting window [fitmin, fitmax] from qmax if unset."""
        qmax = self["qmax"]
        if qmax is not None and not np.isnan(qmax):
            if self["fitmin"] is None or np.isnan(self["fitmin"]):
                self["fitmin"] = float(qmax) / 2.0
            if self["fitmax"] is None or np.isnan(self["fitmax"]):
                self["fitmax"] = float(qmax)

    def set_source(self):
        """Configure standard X-ray laboratory source wavelengths or synchrotron radiation."""
        source_map = {
            "Cu": 1.5418,
            "CuKa1": 1.5405,
            "Ag": 0.5609,
            "AgKa1": 0.5594,
            "Mo": 0.71069,
            "MoKa1": 0.70926,
        }
        src = self["source"]
        if src in source_map:
            self["rambda"] = source_map[src]
            self.set_energy_from_rambda()
            self["polar"] = 0.0
        elif src not in ["synchrotron", "sr"]:
            self.par["source"]["value"] = "synchrotron"

    def tune(self, n_trials=40):
        """Optimize high-Q normalization window to minimize low-r unphysical ripples.

        Uses Optuna Tree-structured Parzen Estimator (TPE) to find optimal
        [fitmin, fitmax] bounds.

        Parameters
        ----------
        n_trials : int, optional
            Number of Optuna search trials. Default is 40.

        Returns
        -------
        PDFget
            Self instance with optimized high-Q window and recomputed G(r).
        """
        from .macro import minimize_ripple

        minimize_ripple.tune(self, n_trials=n_trials)
        self.run()
        return self



    def get_iq(self, data):
        """Converts 2theta to Q and resamples onto a uniform Q grid with Gaussian anti-aliasing."""
        data["q"] = self.twotheta_to_q(data["tt"] + self["zeroshift"])
        data["iq"] = self.noise_filter(data["tt"], data["i"], self["ttfilter"])
        data["q"], data["iq"] = self.twotheta_cut(data["q"], data["iq"])
        return data

    def twotheta_to_q(self, tt):
        """Converts scattering angle 2theta (degrees) to momentum transfer Q (A^-1)."""
        return 4.0 * pi / self["rambda"] * np.sin(np.radians(tt / 2.0))

    def q_to_twotheta(self, q):
        """Converts momentum transfer Q (A^-1) to scattering angle 2theta (degrees)."""
        return np.arcsin(q / (4.0 * pi / self["rambda"])) / pi * 180.0 * 2.0

    def noise_filter(self, x, y, n):
        """Applies an optional FIR low-pass noise filter."""
        if n == 0 or n is None:
            return y
        try:
            z = int(len(x) * (n / x[-1]))
            yf = signal.filtfilt(signal.firwin(z + 1 - z % 2, 10**-8), 1, y)
            return yf
        except Exception:
            return y

    def twotheta_cut(self, q, i):
        """Truncates to [qmin, qmax] and resamples onto a uniform Q grid after anti-aliasing."""
        qmax = float(self["qmax"])
        qmin = float(self["qmin"])

        if qmax >= q[-1]:
            qmax = float(q[-1])

        # Keep a margin around [qmin, qmax] using available raw data
        # so filter boundary effects occur strictly outside [qmin, qmax]
        margin = 0.5
        q_low_bound = max(float(q[0]), qmin - margin)
        q_high_bound = min(float(q[-1]), qmax + margin)
        r = np.where((q >= q_low_bound) & (q <= q_high_bound))
        q_sub = q[r]
        iout = i[r]

        q_fine = q_sub[-1] - q_sub[-2]
        q_finest = np.arange(q_sub[0], q_sub[-1] + q_fine, q_fine)
        iq = np.interp(q_finest, q_sub, iout)

        q_broad = (q_sub[1] - q_sub[0]) / 2.0
        q_ratio = max(q_broad / q_fine, 1.0)
        # Use mode='nearest' so that even if data ends near boundary, it is padded with edge value, not 0
        iq = ndimage.gaussian_filter1d(iq, sigma=q_ratio, mode="nearest")

        qstep = self["qstep"]
        if qstep is None or (isinstance(qstep, float) and np.isnan(qstep)) or qstep <= 0:
            step = q_broad
        else:
            step = float(qstep)

        # Resample on a clean, uniform grid aligning exactly from qmin to qmax
        n_pts = int(np.round((qmax - qmin) / step)) + 1
        q_iq = np.linspace(qmin, qmax, n_pts)
        iq_out = np.interp(q_iq, q_finest, iq)
        return q_iq, iq_out

    def get_i_bg_sub(self, data, background, scale):
        """Subtracts scaled container background intensity in Q-space."""
        if background["tt"].size == 0 or scale == 0.0:
            data["i_bg_sub"] = data["iq"].copy()
        else:
            data["i_bg_sub"] = (
                data["iq"]
                - intp.interp1d(
                    background["q"],
                    background["iq"],
                    "cubic",
                    bounds_error=False,
                    fill_value="extrapolate",
                )(data["q"])
                * scale
            )
        return data

    def get_i_corrected(self, data):
        """Applies Q-region exclusions, capillary absorption, and polarization corrections."""
        data["i_bg_sub"] = self.apply_qexclusion(data["q"], data["i_bg_sub"])
        data["i_bg_sub"] = self.noise_filter(data["q"], data["i_bg_sub"], self["qfilter"])

        data = self.get_absorption(data)
        data = self.get_polarization(data)
        data["i_corrected"] = data["i_bg_sub"] * data["absorption"] / data["polarization"]
        return data

    def apply_qexclusion(self, q, i):
        """Interpolates across excluded Q intervals (e.g., spurious Bragg or detector gaps)."""
        def process(q_arr, i_arr, q_low, q_high):
            idx = np.where((q_arr >= q_low) & (q_arr <= q_high))
            x, y = np.delete(q_arr, idx), np.delete(i_arr, idx)
            y_interp = intp.interp1d(
                x, y, "cubic", bounds_error=False, fill_value="extrapolate"
            )(q_arr)
            i_arr[idx] = y_interp[idx]
            return q_arr, i_arr

        qex = self["qexclude"]
        if qex is None or (isinstance(qex, float) and math.isnan(qex)):
            return i
        q2 = q.copy()
        i2 = i.copy()
        for rng in qex:
            q2, i2 = process(q2, i2, rng[0], rng[1])
        return i2

    def get_absorption(self, data):
        """Evaluates cylindrical capillary absorption correction A_cap(Q, mu*R)."""
        data["absorption"] = self.absorption_correction(
            data["q"], self["rambda"], self["mur"]
        )
        return data

    def get_polarization(self, data):
        """Evaluates 1D beam polarization correction P(Q)."""
        data["polarization"] = self.polarization_correction(data["q"], self)
        return data

    def q_to_twotheta_rad(self, q):
        """Converts momentum transfer Q (A^-1) to scattering angle 2theta (radians)."""
        return np.arcsin(q / (4.0 * pi / self["rambda"])) * 2.0

    def get_form_factor_compton(self, data):
        """Calculates <f^2>, <f>^2, and relativistic Compton scattering I_com(Q)."""
        data["f_sq_ave"], data["f_ave_sq"] = self.get_form_factor(data["q"])
        data["compton"] = self.get_compton(data["q"])
        return data

    def get_form_factor(self, q):
        """Evaluates composition-weighted Faber-Ziman form factor averages <f^2> and <f>^2."""
        return self.form_factor_calculation(
            self["symbol"], self["occ"], q, self["energy"]
        )

    def get_compton(self, q):
        """Evaluates relativistic recoil-attenuated Compton scattering I_com(Q)."""
        cmp, ec_e0 = self.compton_calculation(
            self["symbol"], self["occ"], q, self["energy"]
        )
        return cmp * (ec_e0 ** self["alpha"])

    def get_i_normalized(self, data):
        """Normalizes corrected intensity against theoretical high-Q form factors and Compton scattering."""
        return self.fit_with_form_factor(data)

    def get_base_function(self, data):
        """Constructs the detector/fluorescence baseline basis function B(Q)."""
        def spline_basefit(x, y):
            if self["baseq"] < 5:
                self["baseq"] = 5
            n = int(self["baseq"])
            s = y.size
            m = s // n
            y1 = y[: m * n].reshape(n, -1)
            x1 = x[: m * n].reshape(n, -1)
            r = np.argmin(y1, axis=1)
            r = (tuple(np.arange(n)), tuple(r))
            y1 = y1[r].flatten()
            x1 = x1[r].flatten()
            return intp.interp1d(
                x1, y1, "cubic", bounds_error=False, fill_value="extrapolate"
            )(x)

        def asymmetric_least_squares_smoothing(y, lam, p, niter):
            from scipy import sparse
            from scipy.sparse.linalg import spsolve

            L = y.size
            D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
            D = lam * D.dot(D.transpose())
            w = np.ones(L)
            W = sparse.spdiags(w, 0, L, L)
            for _ in range(niter):
                W.setdiag(w)
                Z = W + D
                z = spsolve(Z, w * y)
                w = p * (y > z) + (1 - p) * (y < z)
            return z

        btype = self["base_function_type"]
        if btype == "data" and data.get("base_function", np.empty(0)).size > 0:
            q = self.twotheta_to_q(data["tt"] + self["zeroshift"])
            _, data["base_function_q"] = self.twotheta_cut(q, data["base_function"])
            data["base_function_q"] = (
                data["base_function_q"] * data["absorption"] / data["polarization"]
            )
            ratio = (
                np.abs(
                    np.average(
                        data["i_corrected"][-20:] / data["base_function_q"][-20:]
                    )
                )
                * 0.5
            )
            data["base_function_q"] = data["base_function_q"] * ratio
        elif btype == "spline":
            data["base_function_q"] = spline_basefit(data["q"], data["i_corrected"])
        elif btype == "alss":
            data["base_function_q"] = asymmetric_least_squares_smoothing(
                data["i_corrected"], 95, 10 ** -int(self["baseq"]), 1
            )
        else:
            # Default to flat baseline if 'flat' or if 'data' was selected without a dummy baseline array
            data["base_function_q"] = (
                np.ones_like(data["q"]) * np.abs(data["i_corrected"][-1]) * 0.5
            )

        return data

    def get_fit_range(self, q):
        """Returns the index mask corresponding to the high-Q fitting window [fitmin, fitmax]."""
        q_high = np.amax([self["fitmin"], self["fitmax"]])
        q_low = np.amin([self["fitmin"], self["fitmax"]])
        q_high = np.amin([q_high, q[-1]])
        q_low = np.amax([q_low, q[0]])
        return np.where((q < q_high) & (q > q_low))

    def fit_with_form_factor(self, data):
        """Fits I_cor(Q) over [fitmin, fitmax] to isolate coherent scattering I_coh(Q)."""
        fit_function = data["f_sq_ave"] + data["compton"]
        r = self.get_fit_range(data["q"])
        fit_scale = data["q"][r]
        data_i = data["i_corrected"][r]
        ff = fit_function[r]

        if self["base_function_type"] not in ["data", "flat", "alss", "spline"]:
            scale, const = fit.autofit_base_none(data_i, ff, fit_scale)
            data["i_normalized"] = (data["i_corrected"] - const) / scale - data["compton"]
            data["base_function_normalized"] = np.zeros_like(data["i_normalized"])
        else:
            base = data["base_function_q"][r]
            scale, bscale, const = fit.autofit_base(data_i, ff, base, fit_scale)
            data["base_function_normalized"] = (
                data["base_function_q"] * bscale + const
            ) / scale
            data["i_normalized"] = (
                data["i_corrected"] / scale
                - data["base_function_normalized"]
                - data["compton"]
            )

        return data

    def get_fq(self, data):
        """Compute Faber-Ziman structure factor S(Q) and reduced structure factor F(Q).

        Parameters
        ----------
        data : dict
            Data dictionary containing normalized intensity and form factor arrays.

        Returns
        -------
        dict
            Updated dictionary containing 'sq', 'fq', and 'q_fq' arrays.
        """
        return self.iq_to_fq(data)

    def get_Gr(self, data):
        """Transform F(Q) into real-space reduced PDF G(r), radial function T(r), and pair distribution g(r).

        Parameters
        ----------
        data : dict
            Data dictionary containing 'q_fq' and 'fq_grid' (or 'fq').

        Returns
        -------
        dict
            Updated dictionary containing 'r', 'Gr', 'gradient', 'tr', and 'gr'.
        """
        fq_for_dst = data["fq_grid"] if "fq_grid" in data else data["fq"]
        data["r"], data["Gr"] = self.dst_transform(
            data["q_fq"], fq_for_dst, self["padding"]
        )
        rho0 = (
            float(self["number_density"])
            if self["number_density"] is not None and self["number_density"] > 0
            else 0.0
        )
        data["gradient"] = -4.0 * pi * rho0 * data["r"]
        # T(r) = 4 * pi * r * rho(r) = G(r) - (-4 * pi * rho0 * r) = G(r) + 4 * pi * rho0 * r
        data["tr"] = data["Gr"] - data["gradient"]

        # g(r) = rho(r) / rho0 = T(r) / (4 * pi * r * rho0) = 1 + G(r) / (4 * pi * r * rho0)
        if rho0 > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                gr = data["tr"] / (4.0 * pi * data["r"] * rho0)
                gr[data["r"] <= 1e-6] = 0.0
                data["gr"] = np.nan_to_num(gr, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            data["gr"] = np.zeros_like(data["r"])

        rmin = float(self["rmin"]) if self["rmin"] is not None else 0.0
        rmax = float(self["rmax"]) if self["rmax"] is not None else float(data["r"][-1])
        mask = (data["r"] >= rmin) & (data["r"] <= rmax)
        data["r"] = data["r"][mask]
        data["Gr"] = data["Gr"][mask]
        data["gradient"] = data["gradient"][mask]
        data["tr"] = data["tr"][mask]
        data["gr"] = data["gr"][mask]
        return data

    def iq_to_fq(self, data):
        """Evaluate S(Q) and F(Q) = Q[S(Q) - 1] on the measured Q grid, and construct
        the extrapolated F(Q) on the Fourier transform grid q_fq."""
        psq = (data["i_normalized"] - data["f_sq_ave"]) / data["f_ave_sq"]
        psq = psq * self.get_revised_lorch(data["q"])

        # 1. Measured experimental data strictly on data["q"] (strictly no extrapolation)
        data["sq"] = psq + 1.0
        data["fq"] = data["q"] * psq

        # 2. Fourier transform grid q_fq from 0.0 to qmax for DST
        qmax = float(self["qmax"])
        if qmax >= data["q"][-1]:
            qmax = float(data["q"][-1])

        step = data["q"][1] - data["q"][0]
        n_fq = int(np.round(qmax / step)) + 1
        data["q_fq"] = np.linspace(0.0, qmax, n_fq)

        # Extrapolated F(Q) array on q_fq: connects (0.0, 0.0) to (q_0, fq_0)
        x = np.append(0.0, data["q"])
        y = np.append(0.0, data["fq"])
        data["fq_grid"] = intp.interp1d(
            x, y, "linear", bounds_error=False, fill_value="extrapolate"
        )(data["q_fq"])
        data["fq_extrapolated"] = data["fq_grid"]

        return data

    def get_revised_lorch(self, q):
        """Evaluate the revised Lorch modification function M(Q).

        Parameters
        ----------
        q : numpy.ndarray
            Momentum transfer grid.

        Returns
        -------
        numpy.ndarray or float
            Lorch window factor array.
        """
        if self["lorch"] > 0.0:
            qmax = float(self["qmax"])
            if qmax >= q[-1]:
                qmax = float(q[-1])
            dlt = pi / qmax / self["lorch"]
            qd = np.where(q * dlt == 0, 1e-12, q * dlt)
            return 3.0 / (qd**3.0) * (np.sin(qd) - qd * np.cos(qd))
        return 1.0

    def dst_transform(self, x, y, pad):
        """Compute Type-I Discrete Sine Transform (DST-I) from F(Q) to G(r).

        Parameters
        ----------
        x : numpy.ndarray
            Evenly spaced momentum transfer grid Q.
        y : numpy.ndarray
            Reduced structure factor F(Q).
        pad : int
            Zero-padding factor (N_padded = N * pad).

        Returns
        -------
        r : numpy.ndarray
            Radial distance grid in angstroms.
        g : numpy.ndarray
            Reduced pair distribution function G(r) in angstroms^-2.
        """
        pad = max(int(pad), 1)
        N = len(x)
        scaling = (1.0 / pi) / float(N) * x[-1]
        g = dst(y, n=N * pad, type=1, norm=None) * scaling
        r = np.arange(1.0, float(N * pad) + 1, 1.0) * pi / (x[-1] * float(pad))
        return r, g

    def idst_transform(self, x, y, pad):
        """Compute inverse Type-I Discrete Sine Transform (iDST-I) from G(r) to F(Q).

        Parameters
        ----------
        x : numpy.ndarray
            Evenly spaced radial distance grid r.
        y : numpy.ndarray
            Reduced pair distribution function G(r).
        pad : int
            Zero-padding factor.

        Returns
        -------
        numpy.ndarray
            Reconstructed F(Q) array.
        """
        pad = max(int(pad), 1)
        N = len(x)
        scaling = (1.0 / pi) / float(N) * x[-1]
        factor = 2.0 * N * pad
        G = y / scaling
        return idst(G, n=N * pad, type=1) / factor

    def get_resolution(self):
        """Return nominal real-space resolution pi / Qmax in angstroms."""
        return pi / self.data["q_fq"][-1]

    def explain(self):
        """Generate a structured Markdown diagnostic summary of the PDF data reduction.

        Returns
        -------
        str
            Multi-line Markdown report covering composition, radiation source,
            Q-ranges, high-Q normalization parameters, and low-r ripple score.
        """
        lines = [
            "# PDF Data Reduction Report (`orochi.pdfget`)",
            f"- **Composition**: `{self['composition']}` (symbols: `{list(self['symbol'])}`, occ: `{list(self['occ'])}`)",
            f"- **Radiation**: source=`{self['source']}`, wavelength=$\\lambda = {self['rambda']:.5f}$ A ($E = {self['energy']:.3f}$ keV)",
            f"- **Momentum Transfer Range**: $Q \\in [{self['qmin']:.2f}, {self['qmax']:.2f}]\\text{{ \\AA}}^{{-1}}$ ($\\Delta Q = {self['qstep']}\\text{{ \\AA}}^{{-1}}$)",
            f"- **High-$Q$ Normalization Window**: $[Q_{{\\mathrm{{fit,min}}}}, Q_{{\\mathrm{{fit,max}}}}] = [{self['fitmin']:.2f}, {self['fitmax']:.2f}]\\text{{ \\AA}}^{{-1}}$",
            f"- **Baseline Mode**: `{self['base_function_type']}`, Revised Lorch width=`{self['lorch']}`, Zero-padding=`{self['padding']}`",
            f"- **Number Density $\\rho_0$**: `{float(self['number_density']):.6f} \\AA^{{-3}}`",
        ]
        if "r" in self.data and "Gr" in self.data:
            r = self.data["r"]
            Gr = self.data["Gr"]
            low_r_mask = r < self["ripple_min_r"]
            if np.any(low_r_mask):
                ripple_score = float(np.mean(np.diff(Gr[low_r_mask], 2) ** 2))
                lines.append(
                    f"- **Sub-atomic Ripple Metric ($r < {self['ripple_min_r']:.2f}$ A)**: `{ripple_score:.4e}`"
                )
        return "\n".join(lines)


def par():
    """Legacy factory returning a new ``PDFget`` instance.

    Returns
    -------
    PDFget
        Newly instantiated PDFget object.
    """
    return PDFget()


def calculator(par_instance=None):
    """Legacy factory returning a configured ``PDFget`` instance.

    Parameters
    ----------
    par_instance : PDFget, optional
        Existing PDFget instance or parameter container. If None, a new
        PDFget instance is created.

    Returns
    -------
    PDFget
        Configured PDFget instance.
    """
    if isinstance(par_instance, PDFget):
        return par_instance
    return PDFget()
