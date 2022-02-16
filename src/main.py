# --------------------------------------------------------------
# |******************** 28/01/2022 ****************************|
# |* Final assignment for Big Data and Business intelligence  *|
# |******************* Federico Canali ************************|
# https://www.kaggle.com/fedesoriano/company-bankruptcy-prediction
# --------------------------------------------------------------
import warnings

import numpy as np
import pandas as pd

import constants as const
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from hyper_parameter_optimization import hyper_parameter_optimization
from modelling import best_model
from model_xgboost import model_xgboost


def get_best_parameters_csv():
    """
    This function takes care of obtaining the best parameters used by the "best_param.csv" file.
    :return: dictionary of parameters
    """
    best = pd.read_csv('best_param.csv', index_col=0)
    dict_best_param = {}

    # convert into records [{...}]
    for i in best.to_dict('records'):
        dict_best_param = dict(i)

    # convert into dictionary and remove parameter nan
    d = dict_best_param.copy()
    for key in dict_best_param:
        if str(dict_best_param[key]) == 'nan' or key not in const.SPACE_X:
            d.pop(key)

    return d


# main
if __name__ == '__main__':
    # change pandas settings to display all rows and columns of a dataset on console
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # remove future warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # read data from csv and store inside a DataFrame
    dataset = pd.read_csv("data.csv")
    # print number of input features (number of features - one target feature)
    print("Number of features: ", len(dataset.columns) - 1)
    # print number of rows inside the dataset
    print("Number of rows: ", len(dataset))

    # check if there are some duplicates row inside dataset
    # and check print the number of duplicates
    print("Number of duplicated rows: ", dataset.duplicated().sum())

    print(" -------------------------------------------------------- ")

    # check if there are some missing values or null values inside dataset
    # and check how many values for each features
    # isnull(): detect missing values for an array-like object.
    # sum(): return the sum of the values over the requested axis.
    print(dataset.isnull().sum())

    print(" -------------------------------------------------------- ")

    # descriptive statistics include those that summarize the central tendency,
    # dispersion and shape of a dataset’s distribution, excluding NaN values.
    print(dataset.describe())

    # plot how each variable is distributed (a graph for each)
    # for feature_name in list(dataset.columns):
    #     dataset.hist(bins=50, column=feature_name)
    #     plt.show()
    # plot how each variable is distributed (a graph for all)
    # dataset.hist(bins=50)
    # plt.show()

    print(" -------------------------------------------------------- ")

    # shows the correlation between the target feature and all the others
    corr = dataset.corr().sort_values(by='Bankrupt?', ascending=False)
    print("\nCorrelation between 'Bankrupt' and other features:\n")
    print(corr['Bankrupt?'])

    print(" -------------------------------------------------------- ")

    # check if the dataset is unbalanced and print the percentage of labels
    # value_count(): return a Series containing counts of unique values.
    labelsCount = dataset['Bankrupt?'].value_counts()
    # "{:.2f}".format() is used to limiting floats to two decimal points
    print("Percentage label 0: ", "{:.2f}".format((labelsCount[0] * 100) / (labelsCount[0] + labelsCount[1])), "%")
    print("Percentage label 1: ", "{:.2f}".format((labelsCount[1] * 100) / (labelsCount[0] + labelsCount[1])), "%")
    print(labelsCount)

    print(" -------------------------------------------------------- ")

    # get all columns except the 'Bankrupt?' (target feature)
    df = dataset.copy()

    # remove also the column 'Net Income Flag' because after standardization it becomes Nan
    X = df.drop(['Bankrupt?', ' Net Income Flag'], axis=1)
    y = df['Bankrupt?']

    # Dived in train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    # ---------------------- Feature Scaling ---------------------------
    avg = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train_scal = (X_train - avg) / std
    # scale also the X_test
    X_test_scal = (X_test - avg) / std

    # visualize X_train after scaling
    print("Input feature after feature scaling: \n")
    print(pd.DataFrame(X_train).describe())

    print(" -------------------------------------------------------- ")

    # ---------------------- Feature Selection ---------------------------
    # Principal component analysis (PCA)
    pca = PCA(n_components=15)
    pca.fit(X_train_scal)
    X_train_reduced = pd.DataFrame(pca.transform(X_train_scal))
    # apply also to the X_test
    X_test_reduced = pd.DataFrame(pca.transform(X_test_scal))
    print("After PCA \n")
    print("Shape of the new matrix X_train: ", pd.DataFrame(X_train_reduced).shape)
    print("Shape of the new matrix X_test: ", pd.DataFrame(X_test_reduced).shape)

    # show distribution of features after PCA
    pd.DataFrame(X_train_reduced).hist(figsize=(30, 30), bins=25)
    plt.show()

    print(" -------------------------------------------------------- ")

    # choose best model
    best_model(X_train_reduced, y_train)

    # True if you want to research the best model parameters.
    # False if you want to load the best parameters found during past searches from the csv file.
    search_best_parameter = True

    # # random forest model
    if search_best_parameter:
        best_param = hyper_parameter_optimization(X_train_reduced, y_train)
    else:
        best_param = get_best_parameters_csv()

    # train and test model
    model_xgboost(X_train_reduced, X_test_reduced, y_train, y_test, best_param)
