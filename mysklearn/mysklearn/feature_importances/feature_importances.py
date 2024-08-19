# importing the packages
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

import seaborn as sns

def feature_importances(data_frame, feature_cols, target_col):
    model_logistic_regression = LogisticRegression(random_state = 42)
    coef_dict = {feature_cols[i]: model_logistic_regression.coef_[0][i] for i in range(len(feature_cols))}

    coef_df = pd.DataFrame(coef_dict.items(), columns=["coef_keys","coef_vals"])
    coef_df["abs_coef_vals"] = np.abs(coef_df["coef_vals"])
    coef_df = coef_df.sort_values(by=["abs_coef_vals"], ascending = False)
    print(coef_df)

    f = plt.figure()
    f.set_figwidth(20)
    f.set_figheight(10)
    sns.barplot(data=coef_df, x= "coef_keys", y="coef_vals" )
    plt.savefig("./mysklearn/feature_importances.png")