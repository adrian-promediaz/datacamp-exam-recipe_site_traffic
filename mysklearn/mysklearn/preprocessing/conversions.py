import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def to_integer(data_frame, column_name):
    data_frame[column_name+"_int"] = data_frame[column_name].str.extract('(\d+)', expand=False).astype(int)
    return data_frame[column_name+"_int"]


def label_encoder(data_frame, column_name):
    labelencorder = LabelEncoder()
    data_frame[column_name+"_categorical"] = labelencorder.fit_transform(data_frame[column_name])
    return data_frame[column_name+"_categorical"]

# recipe_site_traffic_df = pd.read_csv("recipe_site_traffic_2212.csv")

# ## Dropping the features with null values
# feature_cols = ["calories", "carbohydrate", "protein", "sugar"]
# recipe_site_traffic_cleaned_df =  recipe_site_traffic_df.dropna(axis="index", subset=feature_cols)



# columns = ["calories","carbohydrate","sugar","protein"]
# for column in columns:
#     recipe_site_traffic_cleaned_df[column+"_log"] = np.log(recipe_site_traffic_cleaned_df[column])
    # print (recipe_site_traffic_cleaned_df.head())

def remove_outliers(data_frame, column_name):
    globals()[f'q1_{column_name}']= data_frame[column_name].quantile(0.25)
    globals()[f'q3_{column_name}']= data_frame[column_name].quantile(0.75)
    globals()[f'iqr_{column_name}'] = globals()[f'q3_{column_name}'] - globals()[f'q1_{column_name}']
    data_frame = data_frame[~
        (
            (data_frame[column_name] < (globals()[f'q1_{column_name}'] - 1.5 *  globals()[f'iqr_{column_name}'] )) |  (data_frame[column_name] > (globals()[f'q3_{column_name}'] + 1.5 * globals()[f'iqr_{column_name}'])))
        
        ]
    return data_frame[column_name]

# print(recipe_site_traffic_cleaned_df.describe())

## Removing the outliers
# numerical_features_log = ["calories_log","carbohydrate_log","sugar_log","protein_log"]
# for i in numerical_features_log:
#     recipe_site_traffic_cleaned_df[i] = remove_outliers(recipe_site_traffic_cleaned_df,i)
#     recipe_site_traffic_cleaned_df =  recipe_site_traffic_cleaned_df.dropna(axis="index", subset=i)

    # print(recipe_site_traffic_cleaned_df[i].count())
    # print(remove_outliers(recipe_site_traffic_cleaned_df,i))

# print("after outliers are removed")

# print(recipe_site_traffic_cleaned_df.describe())


    