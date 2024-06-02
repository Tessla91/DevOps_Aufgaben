import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import tree, metrics
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import math
import tqdm

# ignore warning messages
import warnings
warnings.filterwarnings("ignore")

# fetch dataset
m_shroom = fetch_ucirepo(id=73)

# data (as pandas dataframes)
x = m_shroom.data.features
y = m_shroom.data.targets

mushroom = pd.concat([x, y], axis=1)

# metadata
print(m_shroom.metadata)

# variable information
print(m_shroom.variables)

# Split data for testing and training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=None)

print('Train set shape: {}'.format(x_train.shape))
print('Test set shape: {}'.format(x_test.shape))

# Rename the column to "target"
y_train.rename(columns={"poisonous": "target"}, inplace=True)
y_test.rename(columns={"poisonous": "target"}, inplace=True)

# Encoding of the target
y_train['target'] = y_train['target'].map({'p': 1, 'e': 0})
y_test['target'] = y_test['target'].map({'p': 1, 'e': 0})

# Label-Encoding of x-values
label_encoders = {}
for column in x_train.columns:
    encoder = LabelEncoder()
    x_train[column] = encoder.fit_transform(x_train[column])
    x_test[column] = encoder.transform(x_test[column])

    # Store the encoder to handle new data or inverse transformation if needed
    label_encoders[column] = encoder

# Normalizing x-values
scaler = MinMaxScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Convert the arrays
x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_test.columns)

x_train = x_train_scaled
x_test = x_test_scaled

# Imputing missing values
kni = KNNImputer ()
x_train = kni.fit_transform(x_train)
x_test = kni.transform(x_test)

x_train = pd.DataFrame(x_train,columns=m_shroom.data.features.columns)
x_test = pd.DataFrame(x_test,columns=m_shroom.data.features.columns)

# dropping "veil-type"
to_drop = ["veil-type"]
x_train.drop(to_drop, axis=1, inplace=True)
x_test.drop(to_drop, axis=1, inplace=True)

# Define feature sets for the four models using the training data
feature_sets = {
    'model_1': x_train.drop(columns=['odor']),
    'model_2': x_train.drop(columns=['odor', 'spore-print-color', 'gill-size', 'gill-spacing', 'population']),
    'model_3': x_train[['odor', 'spore-print-color', 'gill-size', 'gill-spacing', 'population']],
    'model_4': x_train[['odor']]
}

# Define corresponding test feature sets
test_feature_sets = {
    'model_1': x_test.drop(columns=['odor']),
    'model_2': x_test.drop(columns=['odor', 'spore-print-color', 'gill-size', 'gill-spacing', 'population']),
    'model_3': x_test[['odor', 'spore-print-color', 'gill-size', 'gill-spacing', 'population']],
    'model_4': x_test[['odor']]
}

# Define classifiers
classifiers = {
    'kNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(splitter="best", min_samples_split=8, min_samples_leaf=2, max_features=None, max_depth=9, criterion="entropy", class_weight = "balanced"),
    'Logistic Regression': LogisticRegression(solver='lbfgs', penalty='l2', max_iter=200, C=1526.418),
    'SVM': SVC(probability=True, kernel='rbf', gamma=0.0048, degree=5, coef0=3, class_weight=None, C=10000.0),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': RandomForestClassifier (n_estimators=100, min_samples_split=8, min_samples_leaf=8, max_features='sqrt', max_depth=10, criterion='gini', class_weight='balanced_subsample', bootstrap=False),
    'Bagging': VotingClassifier([("knn",KNeighborsClassifier(n_neighbors=5)),("logreg",LogisticRegression(solver='lbfgs', penalty='l2', max_iter=500, C=1526.418)),("random",RandomForestClassifier(n_estimators=100, min_samples_split=8, min_samples_leaf=8, max_features='sqrt', max_depth=10, criterion='gini', class_weight='balanced_subsample', bootstrap=False))], voting='soft')
}

# Initialize results storage
results = []

# Train and evaluate models
for model_name, train_features in feature_sets.items():
    test_features = test_feature_sets[model_name]

    for mushroom_name, mushroom in classifiers.items():
        mushroom.fit(train_features, y_train)
        y_pred = mushroom.predict(test_features)
        y_proba = mushroom.predict_proba(test_features)[:, 1] if hasattr(clf, 'predict_proba') else np.zeros_like(y_pred)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auroc = roc_auc_score(y_test, y_proba) if y_proba.any() else 0

        results.append({
            'Model': model_name,
            'Classifier': mushroom_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'AUROC': auroc
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save to Excel
results_df.to_excel('/Users/Pudzich/Documents/GitHub/Projektarbeit_ML-2024/data/mushroom_results.xlsx', index=False)

print("Results saved to mushroom_results.xlsx")
