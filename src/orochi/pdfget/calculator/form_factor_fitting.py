from scipy.optimize import curve_fit
import numpy as np
import warnings

warnings.simplefilter("ignore")


def autofit_base_none(data, fit_function, fit_scale):

    def func(x, a, c):
        return a * x + c

    a0 = np.abs(np.average(data / fit_function))
    a_min = a0 / 10.0
    a_max = a0 * 10.0

    c0 = 0.0
    c_min = data[-1] * -2.5
    c_max = data[-1] * 2.5

    p1, p2 = curve_fit(
        func,
        fit_function,
        data,
        p0=[a0, c0],
        bounds=([a_min, c_min], [a_max, c_max]),
        maxfev=1000,
        sigma=fit_scale,
    )
    return p1


def autofit_base(data, fit_function, baseline, fit_scale):

    def func(x, a, b, c):
        return a * x + b * baseline + c

    a0 = np.abs(np.average(data / fit_function))
    a_min = a0 / 20.0
    a_max = a0 * 14.0

    b0 = 0.0
    b_min = -np.amax(data) / baseline[-1] / 2.0
    b_max = np.amax(data) / baseline[-1] / 2.0

    c0 = 0.0
    c_min = -np.amax(data) / 2.0
    c_max = np.amax(data) / 2.0
    p1, p2 = curve_fit(
        func,
        fit_function,
        data,
        p0=[a0, b0, c0],
        bounds=([a_min, b_min, c_min], [a_max, b_max, c_max]),
        maxfev=1000,
        sigma=fit_scale,
    )
    return p1


def optuna_base_none(data, fit_function):
    import optuna

    def objective(trial):
        a = trial.suggest_float("a", float(a0 / 2.0), float(a0 * 2.0))
        d = data / a - fit_function
        score = np.average(np.abs(d))
        return float(score)

    a0 = np.abs(np.average(data) / np.average(fit_function))

    optuna.logging.disable_default_handler()
    study = optuna.create_study(
        directions=["minimize"], sampler=optuna.samplers.NSGAIISampler()
    )

    study.enqueue_trial({"a": float(a0)})
    study.optimize(objective, n_trials=30)
    optuna.logging.enable_default_handler()

    return study.best_params["a"]


def optuna_base(data, fit_function, baseline):

    def objective(trial):
        a = trial.suggest_float("a", float(a0 / 2.0), float(a0 * 2.0))
        b = trial.suggest_float("b", float(-b_max), float(b_max))
        d = data / a - fit_function - b * baseline
        score = np.average(np.abs(d))
        return float(score)

    a0 = np.abs(np.average(data) / np.average(fit_function))
    b0 = 1.0
    b_max = 2.0

    optuna.logging.disable_default_handler()
    study = optuna.create_study(
        directions=["minimize"], sampler=optuna.samplers.NSGAIISampler()
    )

    study.enqueue_trial({"a": float(a0), "b": float(b0)})
    study.optimize(objective, n_trials=40)
    optuna.logging.enable_default_handler()

    return study.best_params["a"], study.best_params["b"]
