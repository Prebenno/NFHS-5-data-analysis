# NFHS Data Analysis — Blood Pressure and Hypertension in India

This repository explores blood pressure patterns and hypertension in India using data from NFHS-4 (2015–16) and NFHS-5 (2019–21). The analyses examine how blood pressure varies across age, sex, BMI, height, waist circumference, wealth, education, caste, tobacco use, and self-reported disease status, with an additional analysis on pre- versus post-COVID blood pressure changes.

The project is notebook-driven. Jupyter notebooks load raw DHS Stata files, clean the data, and generate survey-weighted summary statistics and figures.

## What this repository includes

- Survey-weighted means, quantiles, ECDFs, and confidence intervals using the [`balance`](https://import-balance.org/) library
- Comparisons of blood pressure distributions across NFHS-4 and NFHS-5
- Disease-stratified ECDF plots with KS tests (diabetes, heart disease, cancer, CKD, thyroid disease)
- Relative ratios of hypertension stages by education, wealth, caste, and tobacco use
- Analyses of differences between repeated blood pressure measurements and weighted correlations across readings
- Pre- versus post-COVID blood pressure comparisons using time- and state-matched splits, including difference-in-differences heatmaps

## Repository structure

```text
├── main_figs.ipynb              # Main notebook: runs the primary analyses and generates figures
├── covid_analysis.ipynb         # COVID-focused analysis: pre/post splits and DiD heatmaps
├── requirements.txt             # Python dependencies
├── LICENSE
├── src/
│   ├── config.py                # Thresholds, column names, and display constants
│   ├── data_loading.py          # Loaders for NFHS-4/5 household and individual recode DTA files
│   ├── analysis.py              # Weighted statistics, hypertension staging, and ratio calculations
│   └── plotting.py              # Plotting functions
├── data/                        # Placeholder directories; raw data not included
│   ├── household_2021_recode/   # NFHS-5 household member recode (IAPR7EFL.DTA)
│   ├── household_2016/          # NFHS-4 household member recode (IAPR74FL.DTA)
│   ├── individual_2021/         # NFHS-5 women's individual recode (IAIR7EFL.DTA)
│   ├── men_2021/                # NFHS-5 men's recode (IAMR7EFL.DTA)
│   ├── individual_2016/         # NFHS-4 women's individual recode (IAIR74FL.DTA)
│   └── men_2016/                # NFHS-4 men's recode (IAMR74FL.DTA)
