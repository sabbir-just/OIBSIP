# OIBSIP — Data Analytics Track — Task: Predicting House Prices with Linear Regression

## Objective
Build and evaluate a linear regression model that predicts house prices based on features such as
area, location, number of rooms, and age.

## Dataset
- **Source:** [Ames Housing Dataset — Kaggle (shashanknecrothapa/ames-housing-dataset)](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset)
- **File used:** `AmesHousing.csv`

## Tech Stack
- Python, pandas, scikit-learn, matplotlib, seaborn, Jupyter Notebook

## What was done
1. Loaded dataset, checked nulls, and examined the distribution of `SalePrice` (the target).
2. Selected 8 curated predictors (living area, quality, age, basement size, garage, rooms, lot area,
   neighborhood) with documented reasoning, instead of using all ~80 raw columns.
3. Handled missing values (median for numeric, mode for categorical) and one-hot encoded `Neighborhood`.
4. Built a correlation heatmap to confirm the strongest predictors.
5. Split data 80/20 and trained a Linear Regression model.
6. Evaluated with MSE, RMSE, and R².
7. Plotted actual vs. predicted prices and a residual plot.
8. Ran coefficient analysis to identify the strongest positive/negative price drivers.
9. Bonus: compared against Ridge and Lasso regularised models.

## Files in this folder
- `House_Price_Prediction.ipynb` — full notebook
- `house_price_prediction_script.py` — flat Python script version (for quick reference/local runs)
- `README_HousePrices.md` — this file

## How to run
1. Open `House_Price_Prediction.ipynb` in a Kaggle Notebook with the Ames Housing dataset attached,
   or in Colab/Jupyter.
2. If running outside Kaggle, update `DATA_PATH` in Step 1.
3. Run all cells top to bottom.
