import pandas as pd
from matplotlib import pyplot as plt
from model_xgboost import XGBClassifier

import constants as const
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold

def epoch(i, reg):
    """
    Used to return string with epoch and reg_alpha.
    :param i: is the epoch
    :param reg: is the value of reg_alpha
    :return: string
    """

    return str(i) + " (n=" + str(reg) + ")"

def random_search_all_param(X_train, y_train):
    """
    It deals with choosing the best parameters for the model.
    Use the Random search.
    :param X_train: training dataset of feature X
    :param y_train: training dataset of target feature y
    :return: dictionary of the best parameters of the model
    """

    print("@" * 10, " Random Search ", "@" * 10)
    # model
    model = XGBClassifier(use_label_encoder=False)
    num_split = 10
    cv = KFold(n_splits=num_split, random_state=None, shuffle=False)
    # define RandomSearch
    search = RandomizedSearchCV(model, const.SPACE_X, random_state=0, scoring=const.SCORING, n_jobs=-1, cv=cv
                                , return_train_score=True)
    # execute search
    result = search.fit(X_train, y_train)
    # result of the search
    print("Best Score: ", result.best_score_)
    print("Best Hyperparameters: ", result.best_params_)

    print(" ------------------------------------------------------- ")

    # show result to console
    cv_result = pd.DataFrame(result.cv_results_)
    print(cv_result)

    # plot accuracy during training and validation
    epochs = [i for i in range(0, len(cv_result['mean_train_score']))]
    plt.plot(epochs, cv_result['mean_train_score'], 'g', label='Training score')
    plt.plot(epochs, cv_result['mean_test_score'], 'b', label='Validation score')
    plt.title('Training and Validation score')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    print(" ------------------------------------------------------- ")

    return result.best_params_


def number_estimators(X_train, y_train, best_param):
    """
    It deals with choosing the best number of estimators around a range of estimators for the model.
    Use the Grind search.
    :param X_train: training dataset of feature X
    :param y_train: training dataset of target feature y
    :param best_param: dictionary of the best parameters of the model selected by RandomSearch
    :return: dictionary of the best parameters of the model
    """

    print("@" * 10, " Grind Search ", "@" * 10)
    # model
    model = XGBClassifier(use_label_encoder=False)
    # set param to model
    model.set_params(**best_param)
    number_split = 10
    cv = KFold(n_splits=number_split, random_state=None, shuffle=False)
    # search space as a dictionary
    space = dict()
    # Similar accuracy in all numbers of predictors, so I choose the smallest number of predictors
    # and do a RandomSearch around this number
    space['reg_alpha'] = [1e-5, 1e-2, 0.1, 1, 100, 200]
    # define GrindSearch (slowest solution but more accuracy)
    search = GridSearchCV(model, space, n_jobs=-1, cv=cv, scoring=const.SCORING, return_train_score=True)
    # execute search
    result = search.fit(X_train, y_train)
    # result of the search
    print("Best Score: ", result.best_score_)
    print("Best Hyperparameters: ", result.best_params_)

    print(" ------------------------------------------------------- ")

    # show result to console
    cv_result = pd.DataFrame(result.cv_results_)
    print(cv_result)

    # plot accuracy during training and validation
    epochs = [epoch(i,  cv_result['param_reg_alpha'][i]) for i in range(0, len(cv_result['mean_train_score']))]
    plt.plot(epochs, cv_result['mean_train_score'], 'g', label='Training accuracy')
    plt.plot(epochs, cv_result['mean_test_score'], 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    return result.best_params_['reg_alpha']


def hyper_parameter_optimization(X_train, y_train):
    """
    The function is concerned with finding the best parameters for the model.
    :param X_train: training dataset of feature X
    :param y_train: training dataset of target feature y
    :return: dictionary of the best parameters of the model
    """

    best_param = random_search_all_param(X_train, y_train)

    best_param['reg_alpha'] = number_estimators(X_train, y_train, best_param)

    return best_param
