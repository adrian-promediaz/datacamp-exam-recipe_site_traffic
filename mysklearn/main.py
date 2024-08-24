import pandas as pd
import numpy as np
from mysklearn.preprocessing.conversions import to_integer, remove_outliers, label_encoder
from mysklearn.regression.regression import logisitic_regression

recipe_site_traffic_df = pd.read_csv("recipe_site_traffic_2212.csv")

# Converting variables
to_integer(recipe_site_traffic_df, "servings")
categorical_columns = ["high_traffic","category"]
for column in categorical_columns:
    label_encoder(recipe_site_traffic_df, column)
    
## Dropping the features with null values
columns = ["calories", "carbohydrate", "protein", "sugar"]
recipe_site_traffic_cleaned_df =  recipe_site_traffic_df.dropna(axis="index", subset=columns)

## Transforming to logarithmic scale
columns = ["calories","carbohydrate","sugar","protein"]
for column in columns:
    recipe_site_traffic_cleaned_df[column+"_log"] = np.log(recipe_site_traffic_cleaned_df[column])

## Removing outliers
numerical_features_logs = ["calories_log","carbohydrate_log","sugar_log","protein_log"]
for i in numerical_features_logs:
    recipe_site_traffic_cleaned_df[i] = remove_outliers(recipe_site_traffic_cleaned_df,i)
    recipe_site_traffic_cleaned_df =  recipe_site_traffic_cleaned_df.dropna(axis="index", subset=i)

## Performing the regression
feature_cols = ["calories_log", "carbohydrate_log", "protein_log", "sugar_log","servings_int","category_categorical"]
target_col = "high_traffic_categorical"
logisitic_regression(recipe_site_traffic_cleaned_df, feature_cols, target_col)