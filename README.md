# Bayesian-Hierarchical-Analysis-of-Crop-Yield-and-Biomass-in-the-GRACEnet-Data-coding

**STAT 431 Final Project**  
**Authors**: Huixin Li, Weidai He, Qihao Zhang

## Overview

This project performs a **Bayesian hierarchical regression analysis** on the USDA GRACEnet (Greenhouse gas Reduction through Agricultural Carbon Enhancement network) dataset. We model grain carbon content (yield proxy) and above-ground biomass separately using log-transformed responses, with fixed effects for nitrogen input, growing-season weather variables, crop type, tillage, and residue removal, plus random intercepts for experimental sites.

The analysis accounts for site-to-site variability and uses Gibbs sampling for posterior inference.

## Project Structure
Bayesian-Hierarchical-Analysis-of-Crop-Yield-and-Biomass-in-the-GRACEnet-Data-coding/
├── data/
│   └── gracenet_data.csv                 # (Not included - available upon request)
├── outputs/
│   ├── tables/                           # Posterior summaries, descriptive stats
│   └── figures/                          # Exploratory plots, trace plots, site map
├── project_analysis.py                   # Data cleaning, EDA, and summary tables
├── mcmc_hierarchical_model.py            # Gibbs sampler for Bayesian hierarchical model
├── README.md
└── report/
└── Final_Report.pdf                  # Final project report (LaTeX)

## Main Files

- **`project_analysis.py`** — Data preprocessing, exploratory data analysis, descriptive statistics, and preliminary frequentist mixed models.
- **`mcmc_hierarchical_model.py`** — Implements the Bayesian hierarchical model using Gibbs sampling (4 chains, 3500 iterations each).

## Key Findings

- Positive associations: Nitrogen input and growing-season precipitation
- Negative association: Growing-season temperature
- Strong site-level heterogeneity (random intercepts)
- Corn shows substantially higher grain carbon than soybean

## Requirements

- Python 3.12.3
- numpy==1.26.4
- pandas==2.2.2
- matplotlib==3.9.2
- seaborn==0.13.2
- statsmodels==0.14.2

You can install dependencies with:
```bash
pip install numpy pandas matplotlib seaborn statsmodels

## How to Run

```bash
# 1. Run data preparation and EDA
python project_analysis.py

# 2. Run Bayesian MCMC analysis (main results)
python mcmc_hierarchical_model.py
```

## Report

The full project report (PDF) is available in the `report/` folder.

## References

- GRACEnet Dataset: USDA Agricultural Research Service
- Gelman & Hill (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*

