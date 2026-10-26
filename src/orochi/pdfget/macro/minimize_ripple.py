import numpy as np
from scipy.optimize import minimize


def tune(mat_cal, n_trials=40):
    """Optimizes [fitmin, fitmax] to minimize unphysical ripples in G(r) for r < ripple_min_r."""

    def eval_ripple(fmin, fmax):
        mat_cal["fitmin"] = float(fmin)
        mat_cal["fitmax"] = float(fmax)
        mat_cal.data = mat_cal.get_i_normalized(mat_cal.data)
        mat_cal.data = mat_cal.get_fq(mat_cal.data)
        mat_cal.data = mat_cal.get_Gr(mat_cal.data)
        score = np.average(
            np.square(
                np.diff(
                    mat_cal.data["Gr"][
                        mat_cal.data["r"] < mat_cal["ripple_min_r"]
                    ],
                    2,
                )
            )
        )
        return float(score)

    mat_cal.run()
    min0 = float(mat_cal["fitmin"])
    max0 = float(mat_cal["fitmax"])
    q_0 = float(mat_cal.data["q"][0])
    q_max = float(mat_cal.data["q"][-1])

    # Physically sound search bounds for high-Q normalization [fitmin, fitmax]
    min_left = float(np.amin([min0, np.amax([q_0, q_max * 0.45])]))
    min_right = float(np.amax([min0, np.amin([q_max * 0.85, q_max - 2.0])]))
    if min_left >= min_right:
        min_left = float(q_max * 0.45)
        min_right = float(q_max * 0.85)

    max_left = float(np.amax([min_right + 0.5, np.amin([max0, q_max * 0.85])]))
    max_right = float(np.amax([max0, q_max]))
    if max_left >= max_right:
        max_left = float(max_right - 1.0)

    try:
        import optuna

        def objective(trial):
            fmin = trial.suggest_float("fitmin", min_left, min_right)
            fmax = trial.suggest_float("fitmax", max_left, max_right)
            return eval_ripple(fmin, fmax)

        optuna.logging.disable_default_handler()
        study = optuna.create_study(
            directions=["minimize"], sampler=optuna.samplers.TPESampler()
        )
        study.enqueue_trial({"fitmin": min0, "fitmax": max0})
        study.optimize(objective, n_trials=n_trials)
        optuna.logging.enable_default_handler()

        best_fmin = float(study.best_params["fitmin"])
        best_fmax = float(study.best_params["fitmax"])
    except ImportError:
        res = minimize(
            lambda x: eval_ripple(x[0], x[1]),
            x0=[min0, max0],
            bounds=[(min_left, min_right), (max_left, max_right)],
            method="Powell",
            options={"maxfev": n_trials},
        )
        best_fmin, best_fmax = float(res.x[0]), float(res.x[1])

    mat_cal["fitmin"] = best_fmin
    mat_cal["fitmax"] = best_fmax
    mat_cal.run()
    return mat_cal
