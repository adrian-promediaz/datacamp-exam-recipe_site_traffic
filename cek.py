# importing the packages
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

recipe_site_traffic_df = pd.read_csv("recipe_site_traffic_2212.csv")

# Preprocessing Dataset

## Converting the serving feature contains only integer values.
recipe_site_traffic_df["servings_int"] = recipe_site_traffic_df["servings"].str.extract('(\d+)', expand=False)
recipe_site_traffic_df["servings_int"] = recipe_site_traffic_df["servings_int"].astype(int)

## Converting the target variable into a categorical type.
recipe_site_traffic_df["high_traffic_categorical"] = pd.Series(
    np.where(recipe_site_traffic_df["high_traffic"] == "High",1,0), 
    recipe_site_traffic_df.index).astype("category")


## Dropping the features with null values
feature_cols = ["calories", "carbohydrate", "protein", "sugar"]
recipe_site_traffic_cleaned_df =  recipe_site_traffic_df.dropna(axis="index", subset=feature_cols)

## Transforming into logarithmic scale
recipe_site_traffic_cleaned_df["calories_log"] = np.log(recipe_site_traffic_cleaned_df["calories"])
recipe_site_traffic_cleaned_df["carbohydrate_log"] = np.log(recipe_site_traffic_cleaned_df["carbohydrate"])
recipe_site_traffic_cleaned_df["sugar_log"] = np.log(recipe_site_traffic_cleaned_df["sugar"])
recipe_site_traffic_cleaned_df["protein_log"] = np.log(recipe_site_traffic_cleaned_df["protein"])

## Removing the outliers
numerical_features_log = ["calories_log","carbohydrate_log","sugar_log","protein_log"]
for i in numerical_features_log:
    globals()[f'q1_{i}']= recipe_site_traffic_cleaned_df[i].quantile(0.25)
    globals()[f'q3_{i}']= recipe_site_traffic_cleaned_df[i].quantile(0.75)
    globals()[f'iqr_{i}'] = globals()[f'q3_{i}'] - globals()[f'q1_{i}']
    recipe_site_traffic_cleaned_df = recipe_site_traffic_cleaned_df[~
        (
            (recipe_site_traffic_cleaned_df[i] < (globals()[f'q1_{i}'] - 1.5 *  globals()[f'iqr_{i}'] )) |  (recipe_site_traffic_cleaned_df[i] > (globals()[f'q3_{i}'] + 1.5 * globals()[f'iqr_{i}'])))
        
        ]
    
print(recipe_site_traffic_cleaned_df.describe())

# Model Fitting
## Encoding the categorical features
labelencorder = LabelEncoder()
recipe_site_traffic_cleaned_df["servings"] = labelencorder.fit_transform(recipe_site_traffic_cleaned_df["servings"])
recipe_site_traffic_cleaned_df["category_cat"] = labelencorder.fit_transform(recipe_site_traffic_cleaned_df["category"])


## Creating an array containing the features
feature_cols = ["calories_log", "carbohydrate_log", "protein_log", "sugar_log","servings_int","category_cat"]

## Splitting the dataset into train and test sets, with the test size set to 25% of the total dataset, and fixing the random_state
X = recipe_site_traffic_cleaned_df[feature_cols]
y = recipe_site_traffic_cleaned_df["high_traffic_categorical"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

## Logistic Regression
model_logistic_regression = LogisticRegression(random_state = 42)
model_logistic_regression.fit(X_train, y_train)
y_pred_lr = model_logistic_regression.predict(X_test)
y_pred_proba_lr = model_logistic_regression.predict_proba(X_test)[::,1]


conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
print(conf_matrix_lr)

TN = conf_matrix_lr[0,0]
FP = conf_matrix_lr[0,1]
FN = conf_matrix_lr[1,0]
TP = conf_matrix_lr[1,1]

print("precision logistic regression \t: ", round(metrics.precision_score(y_test,y_pred_lr),2))
print("accuracy logistic regression \t: ", round(metrics.accuracy_score(y_test,y_pred_lr),2))
print("recall logistic regression \t\t: ", round(metrics.recall_score(y_test,y_pred_lr),2))
print("f1_score logistic regression \t: ", round(metrics.f1_score(y_test,y_pred_lr),2))

print(classification_report(y_test,y_pred_lr))

# roc_curve
fpr_lr, tpr_lr, _lr = metrics.roc_curve(y_test, y_pred_proba_lr)
auc_lr = metrics.roc_auc_score(y_test, y_pred_proba_lr)
plt.plot(fpr_lr, tpr_lr, label = "auc logistic regression = "+str(round(auc_lr,2)))
plt.legend(loc=4)
# plt.show()

# feature importance

coef_dict = {feature_cols[i]: model_logistic_regression.coef_[0][i] for i in range(len(feature_cols))}

coef_df = pd.DataFrame(coef_dict.items(), columns=["coef_keys","coef_vals"])
coef_df["abs_coef_vals"] = np.abs(coef_df["coef_vals"])
coef_df = coef_df.sort_values(by=["abs_coef_vals"], ascending = False)
print(coef_df)

f = plt.figure()
f.set_figwidth(20)
f.set_figheight(10)
sns.barplot(data=coef_df, x= "coef_keys", y="coef_vals")