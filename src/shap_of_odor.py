import pandas as pd
import numpy as np
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import math
import tqdm
import shap
from shap import Explainer
from shap.plots import beeswarm

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

# no imputing necessary since "odor" and "poisonous" have no missing values

# Rename the column to "target"
y_train.rename(columns={"poisonous": "target"}, inplace=True)
y_test.rename(columns={"poisonous": "target"}, inplace=True)

# Encoding of the target
y_train['target'] = y_train['target'].map({'p': 1, 'e': 0})
y_test['target'] = y_test['target'].map({'p': 1, 'e': 0})

# dropping "veil-type", "cap-shape", "cap-surface", "cap-color", "bruises", "gill-attachment", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-color", "ring-number", "ring-type", "habitat", "spore-print-color", "gill-size", "gill-spacing", "population"
to_drop = ["veil-type", "cap-shape", "cap-surface", "cap-color", "bruises", "gill-attachment", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-color", "ring-number", "ring-type", "habitat", "spore-print-color", "gill-size", "gill-spacing", "population"]
x_train.drop(to_drop, axis=1, inplace=True)
x_test.drop(to_drop, axis=1, inplace=True)

# Encoding of x-values
encoder = OneHotEncoder(drop="first")
encoder.fit(x_train)
x_encoded = encoder.transform(x_train)
x_train = pd.DataFrame(x_encoded.todense(),columns=encoder.get_feature_names_out())

encoder.fit(x_test)
x_encoded2 = encoder.transform(x_test)
x_test = pd.DataFrame(x_encoded2.todense(),columns=encoder.get_feature_names_out())

# Normalizing x- and y-values: not necessary

# Building a Model: KNeirestneighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

# SHAP value for feature importance
print(type(x_train))
print(x_train.shape)

print(type(knn))
print(hasattr(knn, 'predict'))

# Summarize background data using shap.sample
background_samples = shap.sample(x_train, 1)
print(type(background_samples))
print(background_samples.shape)

# SHAP Interpretation
explainer_knn = shap.KernelExplainer(knn.predict, background_samples)
shap_values_knn = explainer_knn.shap_values(x_train)

# Ensure the shapes are as expected
print(f"Background samples shape: {background_samples.shape}")
print(f"SHAP values shape: {np.array(shap_values_knn).shape}")

plt.figure(figsize=(8, 8))
plt.title("SHAP values for kNN")
shap.summary_plot(shap_values_knn, x_train, plot_type="bar")
plt.show()

# Building a Model: Decision Tree
dtc = DecisionTreeClassifier(splitter="best", min_samples_split=8, min_samples_leaf=2, max_features=None, max_depth=9, criterion="entropy", class_weight = "balanced")
dtc.fit(x_train, y_train)

print(type(x_train))
print(x_train.shape)

print(type(dtc))
print(hasattr(dtc, 'predict'))

# Summarize background data using shap.sample
background_samples = shap.sample(x_train, 1)
print(type(background_samples))
print(background_samples.shape)

# SHAP Interpretation
explainer_dtc = shap.KernelExplainer(dtc.predict, background_samples)
shap_values_dtc = explainer_dtc.shap_values(x_train)

# Ensure the shapes are as expected
print(f"Background samples shape: {background_samples.shape}")
print(f"SHAP values shape: {np.array(shap_values_dtc).shape}")

plt.figure(figsize=(8, 8))
plt.title("SHAP values for decision tree")
shap.summary_plot(shap_values_dtc, x_train, plot_type="bar")
plt.show()

# Building a model: Logistic Regression
logreg = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=200, C=1526.418)
logreg.fit(x_train, y_train)

# Analyzing SHAP-value
print(type(x_train))
print(x_train.shape)

print(type(logreg))
print(hasattr(logreg, 'predict'))

# Summarize background data using shap.sample
background_samples = shap.sample(x_train, 1)
print(type(background_samples))
print(background_samples.shape)

# SHAP Interpretation
explainer_logreg = shap.KernelExplainer(logreg.predict, background_samples)
shap_values_logreg = explainer_logreg.shap_values(x_train)

# Ensure the shapes are as expected
print(f"Background samples shape: {background_samples.shape}")
print(f"SHAP values shape: {np.array(shap_values_logreg).shape}")

plt.figure(figsize=(8, 8))
plt.title("SHAP values for logistic regression")
shap.summary_plot(shap_values_logreg, x_train, plot_type="bar")
plt.show()

# Building a model: Support Vector Machine
svm = SVC(probability=True, kernel='rbf', gamma=0.0048, degree=5, coef0=3, class_weight=None, C=10000.0)
svm.fit(x_train, y_train)

# SHAP
print(type(x_train))
print(x_train.shape)

print(type(svm))
print(hasattr(svm, 'predict'))

# Summarize background data using shap.sample
background_samples = shap.sample(x_train, 1)
print(type(background_samples))
print(background_samples.shape)

# SHAP Interpretation
explainer_svm = shap.KernelExplainer(svm.predict, background_samples)
shap_values_svm = explainer_svm.shap_values(x_train)

# Ensure the shapes are as expected
print(f"Background samples shape: {background_samples.shape}")
print(f"SHAP values shape: {np.array(shap_values_svm).shape}")

plt.figure(figsize=(8, 8))
plt.title("SHAP values for support vector machine")
shap.summary_plot(shap_values_svm, x_train, plot_type="bar")
plt.show()

# Building a Model: Random Forest
random = RandomForestClassifier (n_estimators=100, min_samples_split=8, min_samples_leaf=8, max_features='sqrt', max_depth=10, criterion='gini', class_weight='balanced_subsample', bootstrap=False)
random.fit(x_train, y_train)

# Analyzing SHAP-value
print(type(x_train))
print(x_train.shape)

print(type(random))
print(hasattr(random, 'predict'))

# Summarize background data using shap.sample
background_samples = shap.sample(x_train, 1)
print(type(background_samples))
print(background_samples.shape)

# SHAP Interpretation
explainer_random = shap.KernelExplainer(random.predict, background_samples)
shap_values_random = explainer_random.shap_values(x_train)

# Ensure the shapes are as expected
print(f"Background samples shape: {background_samples.shape}")
print(f"SHAP values shape: {np.array(shap_values_random).shape}")

plt.figure(figsize=(8, 8))
plt.title("SHAP values for random forest")
shap.summary_plot(shap_values_random, x_train, plot_type="bar")
plt.show()

# Build a Model: XGBoost
bst = XGBClassifier(subsample=1.0, reg_lambda=0.1, reg_alpha=0.5, n_estimators=100, min_child_weight=3, max_depth=8, learning_rate=0.05, gamma=0.1, colsample_bytree=0.7)
bst.fit(x_train, y_train)

# Analyzing SHAP-value
print(type(x_train))
print(x_train.shape)

print(type(bst))
print(hasattr(bst, 'predict'))

# Summarize background data using shap.sample
background_samples = shap.sample(x_train, 100)
print(type(background_samples))
print(background_samples.shape)

# SHAP Interpretation
explainer_bst = shap.TreeExplainer(bst)
shap_values_bst = explainer_bst.shap_values(x_train)

# Ensure the shapes are as expected
print(f"Background samples shape: {background_samples.shape}")
print(f"SHAP values shape: {np.array(shap_values_bst).shape}")

plt.figure(figsize=(8, 8))
plt.title("SHAP values for XG Boost")
shap.summary_plot(shap_values_bst, x_train, plot_type="bar")
plt.show()

# Building a Model: Bagging-Ensemble
ensemble = VotingClassifier([("knn",KNeighborsClassifier(n_neighbors=5)),("logreg",LogisticRegression(solver='lbfgs', penalty='l2', max_iter=500, C=1526.418)),("random",RandomForestClassifier(n_estimators=100, min_samples_split=8, min_samples_leaf=8, max_features='sqrt', max_depth=10, criterion='gini', class_weight='balanced_subsample', bootstrap=False))], voting='soft')
ensemble.fit(x_train,y_train)

# Analyzing SHAP-value
print(type(x_train))
print(x_train.shape)

print(type(ensemble))
print(hasattr(ensemble, 'predict'))

# Summarize background data using shap.sample
background_samples = shap.sample(x_train, 1)
print(type(background_samples))
print(background_samples.shape)

# SHAP Interpretation
explainer_ensemble = shap.KernelExplainer(ensemble.predict, background_samples)
shap_values_ensemble = explainer_ensemble.shap_values(x_train)

# Ensure the shapes are as expected
print(f"Background samples shape: {background_samples.shape}")
print(f"SHAP values shape: {np.array(shap_values_ensemble).shape}")

plt.figure(figsize=(8, 8))
plt.title("SHAP values for ensemble")
shap.summary_plot(shap_values_ensemble, x_train, plot_type="bar")
plt.show()

