# In this file I define constant variables
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

LOGISTIC_REGRESSION = "Logistic regression"
RANDOM_FOREST = "Random Forest"
ADABOOST = "AdaBoost"
XGBOOST = "XGBoost"
SVC_CLASSIFIER = "SVC"
GAUSSIAN = "GaussianNB"

TARGET_NAMES = ['class 0', 'class 1']

MODELS = {LOGISTIC_REGRESSION: LogisticRegression(),
          RANDOM_FOREST: RandomForestClassifier(),
          ADABOOST: AdaBoostClassifier(),
          XGBOOST: XGBClassifier(use_label_encoder=False),  # in order to remove classifier warnings
          SVC_CLASSIFIER: SVC(),
          GAUSSIAN: GaussianNB()}

MODELS_INDEX = {LOGISTIC_REGRESSION: 0,
                RANDOM_FOREST: 1,
                ADABOOST: 2,
                XGBOOST: 3,
                SVC_CLASSIFIER: 4,
                GAUSSIAN: 5}

SPACE_X = {
    'n_estimators': range(100, 500, 100),
    'max_depth': range(3, 10),
    'min_child_weight': range(1, 6, 2),
    'gamma': [i / 10.0 for i in range(0, 5)],
    'subsample': [i / 10.0 for i in range(6, 10)],
    'colsample_bytree': [i / 10.0 for i in range(6, 10)],
    'scale_pos_weight': [30],  # because of high class imbalance (negative class / positive classes)
    'learning_rate': [0.01]}

SCORING = 'recall'
