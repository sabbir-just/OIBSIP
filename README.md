# OIBSIP — Data Analytics Track — Task: Cleaning Data

## Objective
Demonstrate professional-level data cleaning skills by taking the deliberately messy Titanic dataset
and systematically transforming it into a clean, analysis-ready dataset, with every decision documented.

## Dataset
- **Source:** [Titanic Dataset — Kaggle (yasserh/titanic-dataset)](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- **File used:** `Titanic-Dataset.csv`

## Tech Stack
- Python
- pandas
- numpy
- Jupyter Notebook

## What was done
1. Loaded the dataset and produced an initial data quality report (nulls, duplicates, dtypes).
2. Handled missing values:
   - `Age` → median imputation grouped by `Pclass` and `Sex`
   - `Embarked` → mode imputation
   - `Cabin` → converted to a `Has_Cabin` binary indicator, original column dropped
3. Checked for and removed duplicate rows.
4. Standardised text formatting in `Sex` and `Embarked`.
5. Detected outliers using the IQR method:
   - `Age` outliers retained (biologically valid)
   - `Fare` outliers capped at the upper IQR bound
6. Corrected data types for all columns (categorical, string, float as appropriate).
7. Produced a before-vs-after summary table (row count, null count, duplicate count).
8. Saved the cleaned dataset to `titanic_cleaned.csv`.

## Files in this folder
- `Titanic_Data_Cleaning.ipynb` — full cleaning notebook with markdown explanations
- `titanic_cleaned.csv` — the final cleaned dataset (generated after running the notebook)
- `README.md` — this file

## How to run
1. Open `Titanic_Data_Cleaning.ipynb` in Kaggle Notebooks, Google Colab, or Jupyter.
2. If running outside Kaggle, update `DATA_PATH` in Step 1 to point to your local copy of `Titanic-Dataset.csv`.
3. Run all cells top to bottom.
