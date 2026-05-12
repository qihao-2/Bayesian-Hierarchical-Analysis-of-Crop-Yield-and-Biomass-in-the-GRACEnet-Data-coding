"""
Gibbs sampler for the STAT 431 GRACEnet Bayesian hierarchical models.

The model is a Gaussian hierarchical regression:

    y_i ~ Normal(x_i beta + a_site[i], sigma^2)
    a_j ~ Normal(0, tau^2)
    beta ~ Normal(0, 10^2 I)
    sigma^2 ~ Inverse-Gamma(2, 1)
    tau^2 ~ Inverse-Gamma(2, 1)

Continuous predictors are standardized and categorical predictors are dummy-coded.
The response is log-transformed before fitting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from project_analysis import (
    DATA_PATH,
    TABLE_DIR,
    FIGURE_DIR,
    clean_column_name,
    make_output_dirs,
)


# Model variables
CONTINUOUS_PREDICTORS = ["Total N Amount kgN/ha", "GS_P", "GS_T"]
CATEGORICAL_PREDICTORS = ["Crop", "Tillage Descriptor", "Residue Removal"]
MODEL_COLUMNS = [
    "SiteID",
    "Crop",
    "Tillage Descriptor",
    "Residue Removal",
    "Total N Amount kgN/ha",
    "GS_P",
    "GS_T",
]


# Draw one sample from an inverse-gamma distribution
def inverse_gamma_sample(rng: np.random.Generator, shape: float, scale: float) -> float:
    return 1.0 / rng.gamma(shape=shape, scale=1.0 / scale)


# Prepare the response vector, fixed-effect matrix, and site index
def prepare_model_data(response_col: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(DATA_PATH, na_values=["NA"])
    df["log_grain_c"] = np.log(df["Grain C kgC/ha"])
    df["log_biomass"] = np.log(df["Above G Biomass kg/ha"])

    for col in CONTINUOUS_PREDICTORS:
        df[f"z_{clean_column_name(col)}"] = (df[col] - df[col].mean()) / df[col].std()

    z_columns = [f"z_{clean_column_name(col)}" for col in CONTINUOUS_PREDICTORS]
    data = df[MODEL_COLUMNS + z_columns + [response_col]].dropna().copy()
    y = data[response_col].to_numpy(dtype=float)

    x = pd.DataFrame({"Intercept": np.ones(len(data))}, index=data.index)
    x = pd.concat(
        [
            x,
            data[z_columns],
            pd.get_dummies(data[CATEGORICAL_PREDICTORS], drop_first=True, dtype=float),
        ],
        axis=1,
    )

    site_codes = pd.Categorical(data["SiteID"]).codes
    site_names = list(pd.Categorical(data["SiteID"]).categories)
    return data, y, x.to_numpy(dtype=float), site_codes, list(x.columns), site_names


# Run one Gibbs sampling chain
def run_chain(
    y: np.ndarray,
    x: np.ndarray,
    site_codes: np.ndarray,
    seed: int,
    n_iter: int = 3500,
    burn: int = 1000,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n, p = x.shape
    n_sites = int(site_codes.max()) + 1
    z = np.zeros((n, n_sites))
    z[np.arange(n), site_codes] = 1.0
    w = np.column_stack([x, z])
    k = w.shape[1]

    beta_prior_sd = 10.0
    prior_precision = np.diag(np.r_[np.repeat(1.0 / beta_prior_sd**2, p), np.ones(n_sites)])
    theta = np.zeros(k)
    sigma2 = 0.25
    tau2 = 0.25
    samples: list[np.ndarray] = []

    xtx = w.T @ w
    xty = w.T @ y

    for it in range(n_iter):
        prior_precision[p:, p:] = np.eye(n_sites) / tau2
        precision = xtx / sigma2 + prior_precision
        covariance = np.linalg.inv(precision)
        mean = covariance @ (xty / sigma2)
        theta = rng.multivariate_normal(mean, covariance)

        resid = y - w @ theta
        sigma2 = inverse_gamma_sample(rng, 2.0 + n / 2.0, 1.0 + 0.5 * np.dot(resid, resid))

        site_effects = theta[p:]
        tau2 = inverse_gamma_sample(
            rng,
            2.0 + n_sites / 2.0,
            1.0 + 0.5 * np.dot(site_effects, site_effects),
        )

        if it >= burn:
            samples.append(np.r_[theta[:p], np.sqrt(sigma2), np.sqrt(tau2)])

    return pd.DataFrame(samples)


# Compute the Gelman-Rubin R-hat convergence diagnostic
def rhat(chains: np.ndarray) -> np.ndarray:
    m, n, p = chains.shape
    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)
    between = n * chain_means.var(axis=0, ddof=1)
    within = chain_vars.mean(axis=0)
    var_hat = ((n - 1) / n) * within + between / n
    return np.sqrt(var_hat / within)


# Summarize posterior draws across all chains
def summarize_chains(chains: list[pd.DataFrame], names: list[str]) -> pd.DataFrame:
    arr = np.stack([chain.to_numpy() for chain in chains])
    combined = np.concatenate(arr, axis=0)
    summary = pd.DataFrame(
        {
            "parameter": names,
            "mean": combined.mean(axis=0),
            "sd": combined.std(axis=0, ddof=1),
            "q2.5": np.quantile(combined, 0.025, axis=0),
            "q50": np.quantile(combined, 0.50, axis=0),
            "q97.5": np.quantile(combined, 0.975, axis=0),
            "rhat": rhat(arr),
        }
    )
    return summary


# Plot trace plots for selected parameters
def plot_trace(chains: list[pd.DataFrame], parameter_names: list[str], output_path: Path) -> None:
    fig, axes = plt.subplots(len(parameter_names), 1, figsize=(10, 1.8 * len(parameter_names)), sharex=True)
    if len(parameter_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, parameter_names):
        idx = chains[0].columns.get_loc(name)
        for chain_id, chain in enumerate(chains, start=1):
            ax.plot(chain.iloc[:, idx].to_numpy(), linewidth=0.4, alpha=0.75, label=f"chain {chain_id}")
        ax.set_ylabel(name[:28])
    axes[0].legend(loc="upper right", ncol=4, fontsize=7)
    axes[-1].set_xlabel("post-burn-in iteration")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# Fit the Bayesian hierarchical model for one response
def fit_response(response_col: str, output_prefix: str) -> None:
    _, y, x, site_codes, x_names, _ = prepare_model_data(response_col)
    parameter_names = x_names + ["sigma", "tau_site"]
    chains = [
        run_chain(y, x, site_codes, seed=431 + i).set_axis(parameter_names, axis=1)
        for i in range(4)
    ]

    summary = summarize_chains(chains, parameter_names)
    summary.to_csv(TABLE_DIR / f"{output_prefix}_bayesian_posterior_summary.csv", index=False)

    key_parameters = [
        "Intercept",
        "z_Total_N_Amount_kgN_per_ha",
        "z_GS_P",
        "z_GS_T",
        "sigma",
        "tau_site",
    ]
    key_parameters = [p for p in key_parameters if p in parameter_names]
    plot_trace(chains, key_parameters, FIGURE_DIR / f"{output_prefix}_mcmc_trace.png")


# Main MCMC workflow
def main() -> None:
    make_output_dirs()
    fit_response("log_grain_c", "log_grain_c")
    fit_response("log_biomass", "log_biomass")
    print("MCMC summaries and trace plots written to outputs.")


if __name__ == "__main__":
    main()
