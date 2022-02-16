from matplotlib import pyplot as plt
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
import constants as const
import pandas as pd


def best_model(X, y):
    """
    Decides which is the best model among: Logistic regression, Random Forest, AdaBoost, XGBoost, GaussianNB
    and SVC.
    :param X: training dataset of feature X
    :param y: training dataset of target feature y
    """

    print("@" * 10, " Best model search ", "@" * 10)
    # matrix of all models recall
    recall = [0 for i in range(0, len(const.MODELS))]
    iteration = 0
    num_split = 10
    sfk = KFold(n_splits=num_split, random_state=None, shuffle=False)
    for train_index, validation_index in sfk.split(X, y):
        X_train = pd.DataFrame(X).iloc[train_index]
        X_validation = pd.DataFrame(X).iloc[validation_index]
        y_train = pd.DataFrame(y).iloc[train_index]
        y_validation = pd.DataFrame(y).iloc[validation_index]

        i = 0
        for key in const.MODELS:
            model = const.MODELS[key]
            model.fit(X_train, y_train.values.ravel())
            score = recall_score(y_validation, model.predict(X_validation))
            recall[const.MODELS_INDEX[key]] += score
            print("Recall ", list(const.MODELS_INDEX.keys())[i], " of iteration (", iteration, ") => ", score)
            i += 1
        iteration += 1

    print(" -------------------------------------------------------- ")

    # compute average on each elements in list
    recall = [number / num_split for number in recall]
    # print all recall average
    i = 0
    for r_model in recall:
        print("Recall of ", list(const.MODELS_INDEX.keys())[i], " = ", r_model)
        i += 1

    # print best model
    max_recall = max(recall)
    msg_start = "\nThe best model is "
    msg_end = " with recall of " + str("{:.2f}".format(max_recall * 100)) + "%"
    print(msg_start, list(const.MODELS.keys())[recall.index(max_recall)], msg_end, "\n")

    # plot each model recall
    for i in range(0, len(const.MODELS)):
        # best model bar is green
        if max_recall == recall[i]:
            plt.bar(list(const.MODELS_INDEX.keys())[i], recall[i], color='green')
        else:
            plt.bar(list(const.MODELS_INDEX.keys())[i], recall[i], color='blue')
    plt.title("Models recall")
    plt.xlabel("Models")
    plt.ylabel("Average recall")
    plt.show()
