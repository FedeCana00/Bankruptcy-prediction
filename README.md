![alt python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=whit)

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/FedeCana00/Bankruptcy-prediction/">
    <img src="img/icon.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Bankruptcy prediction</h3>
  
   <p align="center">
   Accurately predicting companies' future failure with Python.
    <br />
    <br />
    <a href="#introduction">Introduction</a>
    ·
    <a href="#data-exploration">Data exploration</a>
    ·
    <a href="#data-visualization">Data visualization</a>
     ·
    <a href="#pre-processing">Pre-processing</a>
     ·
    <a href="#modelling">Modelling</a>
     ·
    <a href="#fine-tuning">Fine tuning</a>
     ·
    <a href="#conclusion">Conclusion</a>
  </p>
</div>

## Introduction

The data was collected by the Taiwan Economic Journal for the years 1999 to 2009. The bankruptcy of the company was defined based on the corporate regulations of the Taiwan Stock Exchange. <br>The goal is to be able to accurately predict the future bankruptcy of companies. The dataset was provided by the <a href="https://www.kaggle.com/fedesoriano/company-bankruptcy-prediction">link</a>.<br>
Contains 95 input and 1 output features. This is a classification task as it tries to predict whether the company will go into bankruptcy (label 1) or not (label 0), or the system is asked to specify which of the k categories an input belongs to.

<br>

## Data exploration

I import the dataset from the csv file to the Pandas software library DataFrame. The DataFrame are two-dimensional tabular data, variable in terms of size, potentially heterogeneous and represent the primary structure of data in Pandas.

```
dataset = pd.read_csv("data.csv")
```

I check the number of input features and the number of data lines contained within the dataset provided.

```
# print number of input features (number of features - one target feature)
print("Number of features: ", len(dataset.columns) - 1)
# print number of rows inside the dataset
print("Number of rows: ", len(dataset))
```

Check that there are no duplicates and missing values.

```
# check if there are some duplicates row inside dataset
# and check print the number of duplicates
print("Number of duplicated rows: ", dataset.duplicated().sum())
# check if there are some missing values or null values inside dataset
# and check how many values for each features
print(dataset.isnull().sum())
```

I analyze the distribution of values for each feature. By running the describe method, I get for each feature: the number of rows, the mean, the standard deviation, the minimum, 25th percentile, 50th percentile, 75th percentile and maximum.

```
# descriptive statistics include those that summarize the central tendency,
# dispersion and shape of a dataset’s distribution, excluding NaN values.
print(dataset.describe())
```

Calculate the correlation of the input features with the target feature. Getting as:
- best five positive correlations:
  - Debt ratio % = 0.250161
  - Current Liability to Assets = 0.194494
  - Borrowing dependency = 0.176543
  - Current Liability to Current Assets = 0.171306
  - Liability to Equity = 0.166812
- best five negative correlations:
  - Net Income to Total Assets = -0.315457
  - ROA(A) before interest and % after tax = -0.282941
  - ROA(B) before interest and depreciation after tax = -0.273051
  - ROA(C) before interest and depreciation before interest = -0.260807
  - Net worth/Assets = -0.250161

```
# shows the correlation between the target feature and all the others
corr = dataset.corr().sort_values(by='Bankrupt?', ascending=False)
print(corr['Bankrupt?'])
```

Then I check if the dataset is balanced or unbalanced, printing on the console the number of occurrences for each label of the output feature.

```
# check if the dataset is unbalanced and print the percentage of labels
# value_count(): return a Series containing counts of unique values.
labelsCount = dataset['Bankrupt?'].value_counts()
# "{:.2f}".format() is used to limiting floats to two decimal points
print("Percentage label 0: ", "{:.2f}".format((labelsCount[0] * 100) /
(labelsCount[0] + labelsCount[1])), "%")
print("Percentage label 1: ", "{:.2f}".format((labelsCount[1] * 100) /
(labelsCount[0] + labelsCount[1])), "%")
print(labelsCount)
```
Output:

```
Percentage label 0: 96.77 %
Percentage label 1: 3.23 %
0 6599
1 220
Name: Bankrupt?, dtype: int64
```
The conclusions I can draw after this analysis are as follows:
- The dataset contains no missing values
- The dataset does not contain duplicate values
- The dataset is made up of numeric values (int64 and float64)
- The dataset is highly unbalanced

## Data visualization

I represent each feature in a histogram to evaluate the distribution of the values of each.

```
# plot how each variable is distributed (a graph for each)
for feature_name in list(dataset.columns):
dataset.hist(bins=50, column=feature_name)
plt.show()
```

## Pre-processing

First I divide the dataset into two DataFrames, one for the input features X and one for the target feature y. Furthermore, I delete the 'Net Income Flag' feature from DataFrame X as the correlation with the target feature is Nan. Then I divide the dataset into 70% training set and 30% test set. I normalize the input features of the test set and of the training set and visualize the change of values on the console.

```
avg = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train_scal = (X_train - avg) / std
# scale also the X_test
X_test_scal = (X_test - avg) / std
# visualize X_train after scaling
print("Input feature after feature scaling: \n")
print(pd.DataFrame(X_train).describe())
```

To perform the feature selection I use Principal component analysis (PCA). It reduces linear dimensionality by Singular Value Decomposition of the data to project it into a lower dimensional space. The number of components to keep set for the PCA is 15. Applies to both training and test input features.

```
# Principal component analysis (PCA)
pca = PCA(n_components=15)
pca.fit(X_train_scal)
X_train_reduced = pd.DataFrame(pca.transform(X_train_scal))
# apply also to the X_test
X_test_reduced = pd.DataFrame(pca.transform(X_test_scal))
```

## Modelling

The models that I have taken into consideration for the comparison are: Logistic regression, Random forest, Adaboost, XGBoost, GaussianNB and SVC. The metric adopted for evaluating the model's performance is recall. This choice depends on the imbalance of the dataset. 
<br>
To train and test the models (with only default parameters) I use KFold as a cross-validator. Provides train / test indexes to divide data into train / test sets. Divide the data set into consecutive k folds. Each fold is then used once as validation while the remaining k - 1 folds form the training set.

```
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
print("Recall ", list(const.MODELS_INDEX.keys())[i], " of iteration (",
iteration, ") => ", score)
i += 1
iteration += 1
```

Where the various models have been evaluated for num_split iterations, the results are shown in a bar graph.
At each run the best model changes between: XGBoost, AdaBoost and GaussianNB. Among them the model I have chosen to use is the XGBoost, which builds an additive model gradually.

## Fine tuning

Now I try to find the best values for parameters of my model. First I run a Random Search with all the parameters I have chosen to test (n_estimators, max_depth, min_child_weight, gamma, subsample, colsample_bytree). I chose recall as a strategy to evaluate the performance of the cross-validated model on the validation set. By setting n_jobs equal to -1 I allow parallel execution on all processors. For the cross-validation split strategy I use the KFold with 10 splits. Also, I set return_train_score to True. The return_train_score is used to get detailed information on how different parameter settings affect the trade-off between overfitting/underfitting.

```
# model
model = RandomForestClassifier()
num_split = 10
cv = StratifiedKFold(n_splits=num_split, random_state=None, shuffle=False)
# define RandomSearch
search = RandomizedSearchCV(model, const.SPACE_RANDOM_FOREST, random_state=0,
scoring='recall', n_jobs=-1, cv=cv
, return_train_score=True)
# execute search
result = search.fit(X_train, y_train)
# result of the search
print("Best Score: ", result.best_score_)
print("Best Hyperparameters: ", result.best_params_)
```

Then I carry out a Grind Search, using the parameters chosen for the model in the previous step. I carry out this research on the parameter reg_alpha: regularization term L1 on weights.
Looking at the graph we notice that when the parameter is less than 1 overfitting occurs.

```
# model
model = XGBClassifier(use_label_encoder=False)
# set param to model
model.set_params(**best_param)
number_split = 10
cv = KFold(n_splits=number_split, random_state=None, shuffle=False)
# search space as a dictionary
space = dict()
# Similar accuracy in all numbers of predictors, so I choose the smallest number
of predictors
# and do a RandomSearch around this number
space['reg_alpha'] = [1e-5, 1e-2, 0.1, 1, 100, 200]
# define GrindSearch (slowest solution but more accuracy)
search = GridSearchCV(model, space, n_jobs=-1, cv=cv, scoring=const.SCORING, return_
train_score=True)
# execute search
result = search.fit(X_train, y_train)
# result of the search
print("Best Score: ", result.best_score_)
print("Best Hyperparameters: ", result.best_params_)
```

In conclusion, I set the previously chosen parameters to my model and train it on the entire training set and run the test on the test set (so far never used).

```
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
```

Output:
```
Accuracy: 81.92 %
Recall score: 90.32 %
Train score: 0.8147915357217683
Test score: 0.8191593352883676
precision recall f1-score support
class 0 1.00 0.82 0.90 1984
class 1 0.13 0.90 0.23 62
accuracy 0.82 2046
macro avg 0.56 0.86 0.56 2046
weighted avg 0.97 0.82 0.88 2046
```

Finally, I draw the confusion matrix and the importance of the features.

## Conclusion

The model can be considered good because although the dataset was strongly unbalanced, it is able to correctly predict the minority class. Looking at the confusion matrix, it can be seen that out of 62 times the label 0 is correctly predicted 56 times and erroneously 6. As regards label 1, it is noted that out of 1984 times it is correctly predicted 1620 times and erroneously 364.
<br>
The recall metric, on the other hand, indicates a value of 90.32%.
