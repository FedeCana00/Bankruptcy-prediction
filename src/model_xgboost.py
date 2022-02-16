import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay\
    , recall_score
from xgboost import XGBClassifier, plot_importance

import constants as const


def write_best_param(param, recall):
    """
    This function takes care of writing the best parameters found in the "best_param.csv" file
    if the recall is better than the past recall.
    :param param: param of model
    :param recall: recall score of model
    """
    param['Recall'] = recall

    try:
        csv = pd.read_csv('best_param.csv')
        csv_param = {}

        # convert into records [{...}]
        for i in csv.to_dict('records'):
            csv_param = dict(i)

        if recall > csv_param['Recall']:
            pd.DataFrame(param, index=[0]).to_csv('best_param.csv')
            print("\n Write into best_param.csv")
    except:
        pd.DataFrame(param, index=[0]).to_csv('best_param.csv')
        print("\n Write into best_param.csv")


def model_xgboost(X_train, X_test, y_train, y_test, best_param):
    """
    Train and test the Random forest model.
    :param X_train: training dataset of feature X
    :param X_test: test dataset of feature X
    :param y_train: training dataset of target feature y
    :param y_test: test dataset of target feature y
    :param best_param: dictionary of the best parameters of the model
    """

    print("@" * 20, " XGBoost ", "@" * 20)
    # define model
    model = XGBClassifier(use_label_encoder=False)
    # set best params found for this model
    model.set_params(**best_param)
    model.fit(X_train, y_train)
    # predict
    y_pred = model.predict(X_test)
    # calculate the accuracy score
    acc = accuracy_score(y_test, y_pred)
    # calculate the recall score
    recall = recall_score(y_test, y_pred)
    print("\nAccuracy: ", "{:.2f}".format(acc * 100), "%")
    print("Recall score: ", "{:.2f}".format(recall * 100), "%\n")
    print("Train score: ", model.score(X_train, y_train))
    print("Test score: ", model.score(X_test, y_test))

    # build a text report showing the main classification metrics
    print(classification_report(y_test, y_pred, target_names=const.TARGET_NAMES))

    # confusion Matrix visualization
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_).plot()
    plt.show()

    # features importance
    plot_importance(model)
    plt.show()

    # memorize best parameters inside a csv file
    write_best_param(model.get_params(), recall)
