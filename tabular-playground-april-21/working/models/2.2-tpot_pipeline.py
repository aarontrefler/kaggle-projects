import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=3)

# Average CV score on the training set was: 0.7816700000000001
exported_pipeline = GradientBoostingClassifier(learning_rate=0.1, max_depth=3, max_features=0.7000000000000001, min_samples_leaf=16, min_samples_split=3, n_estimators=100, subsample=0.4)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 3)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
