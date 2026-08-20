# OIBSIP — Data Analytics Track — Task: EDA on Retail Sales Data

## Objective
Perform a thorough Exploratory Data Analysis on a retail sales dataset to uncover patterns, customer
behaviour trends, and actionable business insights.

## Dataset
- **Source:** [Superstore Dataset (Final) — Kaggle (vivek468/superstore-dataset-final)](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)
- **File used:** `Sample - Superstore.csv`

## Tech Stack
- Python, pandas, matplotlib, seaborn, Jupyter Notebook

## Note on scope
The original task checklist asks for "customer age group and gender breakdown." This dataset does not
include age or gender fields, so **Customer Segment** and **Region** were used instead as the
customer-profile dimensions — this substitution is documented directly in the notebook.

## What was done
1. Loaded and inspected the dataset (shape, dtypes, nulls, duplicates).
2. Descriptive statistics (mean, median, mode, std dev) for Sales, Quantity, Discount, Profit.
3. Time series analysis: monthly and quarterly sales trend line charts.
4. Customer profile analysis: order distribution by Segment and Region.
5. Product analysis: top 10 best-selling products, revenue by category.
6. Correlation heatmap across numeric columns.
7. Extra insight: scatter plot of Profit vs. Discount by Category, revealing where discounting turns
   unprofitable.
8. Conclusion with 3 specific, actionable business recommendations.

## Files in this folder
- `eda-on-retail-sales-data.ipynb` — full analysis notebook
- `superstore_eda_script.py` — same code as a flat, readable Python script (for quick reference/local runs)
- `README_Superstore.md` — this file

## How to run
1. Open `Superstore_EDA.ipynb` in a Kaggle Notebook with the Superstore dataset attached, or in Colab/Jupyter.
2. If running outside Kaggle, update `DATA_PATH` in Step 1 to your local CSV path.
3. Run all cells top to bottom.
