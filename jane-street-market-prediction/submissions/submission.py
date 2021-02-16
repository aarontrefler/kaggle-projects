!pip install datatable==0.11.0 > /dev/null

import datatable as dt
import janestreet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import xgboost as xgb

# Record execution time
start = time.time()

# Preprocessor Utilties
def filter_rows_with_zero_weight(pd_df):
    return pd_df.loc[pd_df.weight != 0]


def fill_missing_values(pd_df):
    return pd_df.fillna(-999)


def create_action_col(pd_df):
    return pd_df.assign(action=
        (((pd_df.weight * pd_df.resp) > 0).astype('int')))                       


def convert_dtypes(pd_df):
    return pd_df.astype(
        {col: np.float32 for col in pd_df.select_dtypes(include='float64').columns})


# Read data
pd_train = dt.fread('/kaggle/input/jane-street-market-prediction/train.csv').to_pandas()

# Preprocess data
pd_train_clean = (pd_train
    .pipe(filter_rows_with_zero_weight)
    .pipe(fill_missing_values)
    .pipe(create_action_col)
    .pipe(convert_dtypes))
del pd_train


# Train setup
clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=11,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.7,
    missing=-999,
    random_state=2020,
    tree_method='gpu_hist',
    use_label_encoder=False)
X = pd_train_clean.filter(like="feature")
y = pd_train_clean.action


# Train
clf.fit(X, y)

# Submission
env = janestreet.make_env()  # initialize the environment
iter_test = env.iter_test()  # an iterator which loops over the test set
for (test_df, pred_df) in iter_test:
    if test_df.weight.item() > 0:
        X = test_df.filter(like="feature")
        X = X.fillna(-999)
        pred = clf.predict(X)[0]
        pred_df.action = pred
    else:
        pred_df.action = 0
    env.predict(pred_df)
    
# Showcase execution time
print("{:.2f}min".format((time.time() - start) / 60))