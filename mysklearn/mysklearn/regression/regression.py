from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from mdutils.mdutils import MdUtils
from mdutils import Html

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def logisitic_regression(data_frame, feature_cols, target_col):
    X = data_frame[feature_cols]
    y = data_frame[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)
    
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
    print("accuracy logistic regression \t\t: ", round(metrics.accuracy_score(y_test,y_pred_lr),2))
    print("recall logistic regression \t\t: ", round(metrics.recall_score(y_test,y_pred_lr),2))
    print("f1_score logistic regression \t\t: ", round(metrics.f1_score(y_test,y_pred_lr),2))

    class_report = classification_report(y_test,y_pred_lr)
    print(class_report)

    # roc_curve
    fpr_lr, tpr_lr, _lr = metrics.roc_curve(y_test, y_pred_proba_lr)
    auc_lr = metrics.roc_auc_score(y_test, y_pred_proba_lr)
    plt.plot(fpr_lr, tpr_lr, label = "auc logistic regression = "+str(round(auc_lr,2)))
    plt.legend(loc=4)
    plt.savefig("./mysklearn/logistic_regression_auc.png")

    md_file = MdUtils(file_name="./mysklearn/logistic_regression_results", title ="logistic_regression_results")
    md_file.new_paragraph(f"precision logistic regression \t: {round(metrics.precision_score(y_test,y_pred_lr),2)}")
    md_file.new_paragraph(f"accuracy logistic regression \t: {round(metrics.accuracy_score(y_test,y_pred_lr),2)}")
    md_file.new_paragraph(f"recall logistic regression \t: {round(metrics.recall_score(y_test,y_pred_lr),2)}")
    md_file.new_paragraph(f"f1_score logistic regression \t: {round(metrics.f1_score(y_test,y_pred_lr),2)}")
    
    md_file.new_paragraph(f"\n{class_report}\n")
    md_file.create_md_file()

    coef_dict = {feature_cols[i]: model_logistic_regression.coef_[0][i] for i in range(len(feature_cols))}

    coef_df = pd.DataFrame(coef_dict.items(), columns=["coef_keys","coef_vals"])
    coef_df["abs_coef_vals"] = np.abs(coef_df["coef_vals"])
    coef_df = coef_df.sort_values(by=["abs_coef_vals"], ascending = False)
    print(coef_df)

    f = plt.figure()
    f.set_figwidth(20)
    f.set_figheight(10)

    sns.barplot(data=coef_df, x= "coef_keys", y="coef_vals" )
    plt.savefig("./mysklearn/logistinc_regression_feature_importances.png")
