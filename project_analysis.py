"""
Project: Bayesian hierarchical analysis of GRACEnet crop outcomes.

This script performs the following steps:
1. Define and document variables used in the analysis.
2. Report data processing decisions, geographic/temporal extent, and site sizes.
3. Produce descriptive summaries and exploratory figures.
4. Fit preliminary mixed-effect models.
5. Provide a PyMC Bayesian hierarchical model template for MCMC.

Run:
    python project_analysis.py

Outputs are written to:
    outputs/tables/
    outputs/figures/
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# Project paths and output locations
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "gracenet_data.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"


# Variable documentation used in the report
VARIABLE_DESCRIPTIONS = pd.DataFrame(
    [
        {
            "variable": "Grain C kgC/ha",
            "role": "primary response",
            "meaning": "grain carbon content, used as the main crop yield outcome",
            "unit": "kg C / ha",
        },
        {
            "variable": "Above G Biomass kg/ha",
            "role": "secondary response",
            "meaning": "above-ground crop biomass",
            "unit": "kg / ha",
        },
        {
            "variable": "SiteID",
            "role": "grouping variable",
            "meaning": "experimental site; used as a random intercept group",
            "unit": "site label",
        },
        {
            "variable": "Crop",
            "role": "categorical predictor",
            "meaning": "crop species, mostly corn and soybean in this data subset",
            "unit": "category",
        },
        {
            "variable": "Tillage Descriptor",
            "role": "categorical predictor",
            "meaning": "tillage management class",
            "unit": "category",
        },
        {
            "variable": "Residue Removal",
            "role": "categorical predictor",
            "meaning": "whether crop residue was removed",
            "unit": "category",
        },
        {
            "variable": "Total N Amount kgN/ha",
            "role": "continuous predictor",
            "meaning": "annual total nitrogen fertilizer application amount",
            "unit": "kg N / ha",
        },
        {
            "variable": "GS_P",
            "role": "continuous predictor",
            "meaning": "total precipitation during the growing season",
            "unit": "mm",
        },
        {
            "variable": "GS_T",
            "role": "continuous predictor",
            "meaning": "average temperature during the growing season",
            "unit": "degrees C",
        },
        {
            "variable": "MAP mm",
            "role": "descriptive/site climate variable",
            "meaning": "mean annual precipitation",
            "unit": "mm",
        },
        {
            "variable": "MAT degC",
            "role": "descriptive/site climate variable",
            "meaning": "mean annual temperature",
            "unit": "degrees C",
        },
        {
            "variable": "Organic C gC/kg",
            "role": "secondary soil predictor",
            "meaning": "soil organic carbon content",
            "unit": "g C / kg",
        },
        {
            "variable": "pH",
            "role": "secondary soil predictor",
            "meaning": "soil pH",
            "unit": "pH scale",
        },
        {
            "variable": "Sand %",
            "role": "secondary soil predictor",
            "meaning": "soil sand proportion",
            "unit": "percent",
        },
        {
            "variable": "Clay %",
            "role": "secondary soil predictor",
            "meaning": "soil clay proportion",
            "unit": "percent",
        },
    ]
)


# Core columns and predictors for the analysis
BASE_MODEL_COLUMNS = [
    "SiteID",
    "Crop",
    "Tillage Descriptor",
    "Residue Removal",
    "Grain C kgC/ha",
    "Above G Biomass kg/ha",
    "Year",
    "Total N Amount kgN/ha",
    "GS_P",
    "GS_T",
    "MAP mm",
    "MAT degC",
]

CONTINUOUS_PREDICTORS = ["Total N Amount kgN/ha", "GS_P", "GS_T"]


# Create output folders
def make_output_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# Convert raw column names into code-friendly names
def clean_column_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("/", "_per_")
        .replace("%", "pct")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )


# Load the CSV file and create transformed model variables
def load_and_prepare_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, na_values=["NA"])

    for col in ["SiteID", "Crop", "Tillage Descriptor", "Residue Removal"]:
        df[col] = df[col].astype("category")

    # Positive, right-skewed agricultural outcomes are modeled on the log scale.
    df["log_grain_c"] = np.log(df["Grain C kgC/ha"])
    df["log_biomass"] = np.log(df["Above G Biomass kg/ha"])

    for col in CONTINUOUS_PREDICTORS:
        mean = df[col].mean(skipna=True)
        std = df[col].std(skipna=True)
        df[f"z_{clean_column_name(col)}"] = (df[col] - mean) / std

    return df


# Write data documentation, missingness, site, and summary tables
def write_project_metadata(df: pd.DataFrame) -> None:
    VARIABLE_DESCRIPTIONS.to_csv(TABLE_DIR / "variable_descriptions.csv", index=False)

    missing = (
        df.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_percent=lambda x: 100 * x["missing_count"] / len(df))
        .sort_values("missing_percent", ascending=False)
    )
    missing.to_csv(TABLE_DIR / "missingness_summary.csv")

    site_summary = (
        df.groupby("SiteID", observed=True)
        .agg(
            n_observations=("SiteID", "size"),
            first_year=("Year", "min"),
            last_year=("Year", "max"),
            mean_grain_c=("Grain C kgC/ha", "mean"),
            mean_biomass=("Above G Biomass kg/ha", "mean"),
            mean_gs_p=("GS_P", "mean"),
            mean_gs_t=("GS_T", "mean"),
        )
        .sort_values("n_observations", ascending=False)
    )
    site_summary.to_csv(TABLE_DIR / "site_summary.csv")

    extent = pd.DataFrame(
        [
            {
                "n_rows": len(df),
                "n_sites": df["SiteID"].nunique(),
                "first_year": int(df["Year"].min()),
                "last_year": int(df["Year"].max()),
                "median_observations_per_site": site_summary["n_observations"].median(),
                "min_observations_per_site": site_summary["n_observations"].min(),
                "max_observations_per_site": site_summary["n_observations"].max(),
            }
        ]
    )
    extent.to_csv(TABLE_DIR / "data_extent_summary.csv", index=False)

    descriptive = df[
        [
            "Grain C kgC/ha",
            "Above G Biomass kg/ha",
            "Total N Amount kgN/ha",
            "GS_P",
            "GS_T",
            "MAP mm",
            "MAT degC",
            "Organic C gC/kg",
            "pH",
            "Sand %",
            "Clay %",
        ]
    ].describe()
    descriptive.to_csv(TABLE_DIR / "numeric_descriptive_summary.csv")

    for col in ["Crop", "Tillage Descriptor", "Residue Removal"]:
        counts = df[col].value_counts(dropna=False).rename_axis(col).reset_index(name="n")
        counts.to_csv(TABLE_DIR / f"{clean_column_name(col)}_counts.csv", index=False)


# Create exploratory figures for the report
def make_figures(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df["Grain C kgC/ha"], kde=True, ax=axes[0])
    axes[0].set_title("Grain C")
    sns.histplot(df["log_grain_c"], kde=True, ax=axes[1])
    axes[1].set_title("Log Grain C")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "grain_c_distribution.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df["Above G Biomass kg/ha"], kde=True, ax=axes[0])
    axes[0].set_title("Above-Ground Biomass")
    sns.histplot(df["log_biomass"], kde=True, ax=axes[1])
    axes[1].set_title("Log Above-Ground Biomass")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "biomass_distribution.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=df, x="SiteID", y="log_grain_c", ax=ax)
    ax.set_title("Log Grain C by Site")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "log_grain_c_by_site.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="GS_P",
        y="Grain C kgC/ha",
        hue="Crop",
        alpha=0.55,
        ax=ax,
    )
    ax.set_title("Grain C vs. Growing-Season Precipitation")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "grain_c_vs_gs_p.png", dpi=300)
    plt.close(fig)


# Build a complete-case model data set for one response
def model_dataset(df: pd.DataFrame, response: str) -> pd.DataFrame:
    columns = [
        "SiteID",
        "Crop",
        "Tillage Descriptor",
        "Residue Removal",
        response,
        "z_Total_N_Amount_kgN_per_ha",
        "z_GS_P",
        "z_GS_T",
    ]
    data = df[columns].dropna().copy()
    for col in ["SiteID", "Crop", "Tillage Descriptor", "Residue Removal"]:
        data[col] = data[col].cat.remove_unused_categories()
    return data


# Fit a preliminary frequentist mixed-effect model
def fit_preliminary_mixed_model(df: pd.DataFrame, response: str, output_name: str) -> None:
    data = model_dataset(df, response)
    formula = (
        f"{response} ~ z_Total_N_Amount_kgN_per_ha + z_GS_P + z_GS_T "
        "+ C(Crop) + C(Q('Tillage Descriptor')) + C(Q('Residue Removal'))"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(formula, data=data, groups=data["SiteID"])
        result = model.fit(reml=False, method="lbfgs", maxiter=1000)

    (TABLE_DIR / f"{output_name}_mixedlm_summary.txt").write_text(
        result.summary().as_text(),
        encoding="utf-8",
    )


# Optional PyMC model template if PyMC and ArviZ are installed
def fit_optional_bayesian_model(df: pd.DataFrame) -> None:
    """
    Bayesian hierarchical model template.

    The final report should describe this model mathematically, for observation i
    in site j[i]:

        y_i ~ Normal(mu_i, sigma)
        mu_i = alpha + a_site[j[i]] + x_i beta
        a_site[j] ~ Normal(0, sigma_site)
        alpha ~ Normal(mean(log grain C), 2)
        beta_k ~ Normal(0, 1)
        sigma ~ HalfNormal(1)
        sigma_site ~ HalfNormal(1)

    y_i is log(Grain C kgC/ha). Continuous predictors are standardized.
    Categorical predictors are dummy-coded.

    This function runs only if pymc and arviz are installed.
    """
    try:
        import pymc as pm
        import arviz as az
    except ImportError:
        message = (
            "PyMC/ArviZ are not installed in this environment, so the Bayesian "
            "MCMC model was not run. Install them with conda or pip, then call "
            "fit_optional_bayesian_model(df) from this script."
        )
        (TABLE_DIR / "bayesian_model_not_run.txt").write_text(message, encoding="utf-8")
        print(message)
        return

    data = model_dataset(df, "log_grain_c")
    y = data["log_grain_c"].to_numpy()
    site_codes = data["SiteID"].cat.codes.to_numpy()
    n_sites = data["SiteID"].nunique()

    x = pd.get_dummies(
        data[
            [
                "z_Total_N_Amount_kgN_per_ha",
                "z_GS_P",
                "z_GS_T",
                "Crop",
                "Tillage Descriptor",
                "Residue Removal",
            ]
        ],
        drop_first=True,
        dtype=float,
    )
    x_values = x.to_numpy()

    with pm.Model() as hierarchical_model:
        alpha = pm.Normal("alpha", mu=float(np.mean(y)), sigma=2.0)
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=x_values.shape[1])
        sigma_site = pm.HalfNormal("sigma_site", sigma=1.0)
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        site_offset = pm.Normal("site_offset", mu=0.0, sigma=1.0, shape=n_sites)
        site_effect = pm.Deterministic("site_effect", site_offset * sigma_site)

        mu = alpha + site_effect[site_codes] + pm.math.dot(x_values, beta)
        pm.Normal("log_grain_c_obs", mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(
            draws=2000,
            tune=1000,
            chains=4,
            target_accept=0.9,
            random_seed=431,
        )
        posterior_predictive = pm.sample_posterior_predictive(
            idata,
            random_seed=431,
        )

    summary = az.summary(
        idata,
        var_names=["alpha", "beta", "sigma", "sigma_site"],
        round_to=3,
    )
    summary.to_csv(TABLE_DIR / "pymc_log_grain_c_posterior_summary.csv")
    az.to_netcdf(idata, TABLE_DIR / "pymc_log_grain_c_trace.nc")
    az.to_netcdf(posterior_predictive, TABLE_DIR / "pymc_log_grain_c_ppc.nc")


# Main analysis workflow
def main() -> None:
    make_output_dirs()
    df = load_and_prepare_data()

    write_project_metadata(df)
    make_figures(df)

    fit_preliminary_mixed_model(df, "log_grain_c", "log_grain_c")
    fit_preliminary_mixed_model(df, "log_biomass", "log_biomass")

    # Leave this enabled: it will run if PyMC/ArviZ are installed and otherwise
    # write a short note explaining why Bayesian MCMC was skipped.
    fit_optional_bayesian_model(df)

    print("Analysis complete.")
    print(f"Tables: {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
