import pandas as pd
import numpy as np
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import math
import tqdm
from mpl_toolkits.mplot3d import Axes3D
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

# Imputing missing values with pandas dataframe
si = SimpleImputer(strategy='most_frequent')
x_train = si.fit_transform(x_train)
x_test = si.transform(x_test)

x_train = pd.DataFrame(x_train,columns=m_shroom.data.features.columns)
x_test = pd.DataFrame(x_test,columns=m_shroom.data.features.columns)

# no missing target-values

# Rename the column to "target"
y_train.rename(columns={"poisonous": "target"}, inplace=True)
y_test.rename(columns={"poisonous": "target"}, inplace=True)

# Encoding of the target
y_train['target'] = y_train['target'].map({'p': 1, 'e': 0})
y_test['target'] = y_test['target'].map({'p': 1, 'e': 0})

# Encoding of x-values
encoder = OneHotEncoder(drop="first")
encoder.fit(x_train)
x_encoded = encoder.transform(x_train)
x_train = pd.DataFrame(x_encoded.todense(),columns=encoder.get_feature_names_out())

encoder.fit(x_test)
x_encoded2 = encoder.transform(x_test)
x_test = pd.DataFrame(x_encoded2.todense(),columns=encoder.get_feature_names_out())

# Normalizing x- and y-values: not necessary

# Multicollinearity

# Reset indices of x_train and y_train
x_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)

# Concatenate x_train and y_train into mushroom_train DataFrame
mushroom_train = pd.concat([x_train, y_train], axis=1)

corr_matrix = mushroom_train.corr().abs()
corr_matrix

# Filter out correlations where both features are the same (correlation equals 1)
mask = np.eye(len(corr_matrix), dtype=bool)
filtered_corr = corr_matrix.mask(mask)

# Find correlations greater than or equal to 0.5
high_corr = filtered_corr[filtered_corr >= 0.5].stack().reset_index()

high_corr.head (20)

# Building a Model: KNeirestneighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

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

# calculate accuracy, precision and recall for knn
predict_knn = knn.predict(x_test)

cm = confusion_matrix(predict_knn, y_test)
_, fp, fn, tp = cm.ravel()

print("accuracy for knn: {}".format(accuracy_score(predict_knn, y_test)))
print("precision for knn: {}".format(tp/(tp + fp)))
print("recall for knn: {}".format(tp/(tp + fn)))

# Building a Model: Decision Tree
dtc = DecisionTreeClassifier(max_depth=4, min_samples_split=50, class_weight = "balanced", random_state=1)
dtc.fit(x_train, y_train)

fig = plt.figure(figsize=(15,15))
baum = tree.plot_tree(dtc, filled=True)

# calculate feature importance with SHAP
explainer_short_random = shap.Explainer(random)
sv_dtc= explainer_short_random(x_train)

shap.plots.beeswarm(sv_dtc[:, :, 1])

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
plt.title("SHAP values for logistic regression")
shap.summary_plot(shap_values_dtc, x_train, plot_type="bar")
plt.show()

# calculate accuracy, precision and recall for dtc

predict_dtc = dtc.predict(x_test)

cm = confusion_matrix(predict_dtc, y_test)
_, fp, fn, tp = cm.ravel()

print("accuracy for Decision Tree: {}".format(accuracy_score(predict_dtc, y_test)))
print("precision for Decision Tree: {}".format(tp/(tp + fp)))
print("recall for Decision Tree: {}".format(tp/(tp + fn)))

# Building a model: Logistic Regression
logreg = LogisticRegression()
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

# calculate accuracy, precision and recall for logistic regression
predict_logreg = logreg.predict(x_test)

cm = confusion_matrix(predict_logreg, y_test)
_, fp, fn, tp = cm.ravel()

print("accuracy for Logistic Regression: {}".format(accuracy_score(predict_logreg, y_test)))
print("precision for Logistic Regression: {}".format(tp/(tp + fp)))
print("recall for Logistic Regression: {}".format(tp/(tp + fn)))

# Building a model: Support Vector Machine
svm = SVC(probability=True)
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
plt.title("SHAP values for logistic regression")
shap.summary_plot(shap_values_svm, x_train, plot_type="bar")
plt.show()

# calculate accuracy, precision and recall for support vector machine
predict_svm = svm.predict(x_test)

cm = confusion_matrix(predict_svm, y_test)
_, fp, fn, tp = cm.ravel()
print("accuracy of support vector machine: {}".format(accuracy_score(predict_svm, y_test)))
print("precision of support vectir machine: {}".format(tp/(tp + fp)))
print("recall of support vector machine: {}".format(tp/(tp + fn)))

# Building a Model: Random Forest
random = RandomForestClassifier (criterion="entropy", max_depth=4, min_samples_split=50, random_state=1)
random.fit(x_train, y_train)

# calculate feature importance with SHAP
explainer_short_random = shap.Explainer(random)
sv_random = explainer_short_random(x_train)

shap.plots.beeswarm(sv_random[:, :, 1])

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

# calculate accuracy, precision and recall for random forest
predict_random = random.predict(x_test)

cm = confusion_matrix(predict_random, y_test)
_, fp, fn, tp = cm.ravel()

print("accuracy of random forest: {}".format(accuracy_score(predict_random, y_test)))
print("precision of random forest: {}".format(tp/(tp + fp)))
print("recall of random forest: {}".format(tp/(tp + fn)))

# Build a Model: XGBoost
bst = XGBClassifier(subsample=0.6, reg_lambda=0, reg_alpha=0, n_estimators=100, min_child_weight=1, max_depth=6, learning_rate=0.1, gamma=0.1, colsample_bytree=0.6, objective='binary:logistic')
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

#explainer_bst = shap.KernelExplainer(bst.predict, background_samples)
#shap_values_bst = explainer_bst.shap_values(x_train)

# Ensure the shapes are as expected
print(f"Background samples shape: {background_samples.shape}")
print(f"SHAP values shape: {np.array(shap_values_bst).shape}")

plt.figure(figsize=(8, 8))
plt.title("SHAP values for XG Boost")
shap.summary_plot(shap_values_bst, x_train, plot_type="bar")
plt.show()

# calculate accuracy, precision and recall for XG boost
predict_bst= bst.predict(x_test)

cm = confusion_matrix(predict_bst, y_test)
_, fp, fn, tp = cm.ravel()

print("accuracy of XG Boost: {}".format(accuracy_score(predict_bst, y_test)))
print("precision of XG Boost: {}".format(tp/(tp + fp)))
print("recall of XG Boost: {}".format(tp/(tp + fn)))

# Building a Model: Bagging-Ensemble
ensemble = VotingClassifier([("knn",KNeighborsClassifier()),("logreg",LogisticRegression()),("random",RandomForestClassifier(criterion="entropy", max_depth=5, min_samples_split=5, random_state=1))], voting='soft')
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

# calculate accuracy, precision and recall for ensemble
predict_ensemble= ensemble.predict(x_test)

cm = confusion_matrix(predict_ensemble, y_test)
_, fp, fn, tp = cm.ravel()

print("accuracy of ensemble: {}".format(accuracy_score(predict_ensemble, y_test)))
print("precision of ensemble: {}".format(tp/(tp + fp)))
print("recall of ensemble: {}".format(tp/(tp + fn)))

# Graphing all ROC-Curves in one
plt.figure(figsize=(8,6))
plt.title("ROC Curve")
fpr, tpr, thresholds = metrics.roc_curve(y_test.values, ensemble.predict_proba(x_test)[:,1:])
fpr1, tpr1, thresholds1 = metrics.roc_curve(y_test.values, random.predict_proba(x_test)[:,1:])
fpr2, tpr2, thresholds2 = metrics.roc_curve(y_test.values, logreg.predict_proba(x_test)[:,1:])
fpr3, tpr3, thresholds3 = metrics.roc_curve(y_test.values, dtc.predict_proba(x_test)[:,1:])
fpr4, tpr4, thresholds4 = metrics.roc_curve(y_test.values, knn.predict_proba(x_test)[:,1:])
fpr5, tpr5, thresholds5 = metrics.roc_curve(y_test.values, svm.predict_proba(x_test)[:,1:])
fpr6, tpr6, thresholds6 = metrics.roc_curve(y_test.values, bst.predict_proba(x_test)[:,1:])
plt.plot([0,1],[0,1],ls="--",c="white",alpha=0.2)
plt.plot(fpr,tpr,label="ROC-Graph Ensemble",c="#1ACC94")
plt.plot(fpr1,tpr1,label="ROC-Graph Random Forest",c="yellow")
plt.plot(fpr2,tpr2,label="ROC-Graph Logistic Regression",c="blue")
plt.plot(fpr3,tpr3,label="ROC-Graph Decision Tree",c="red")
plt.plot(fpr4,tpr4,label="ROC-Graph kNN",c="purple")
plt.plot(fpr5,tpr5,label="ROC-Graph Support Vector Machine",c="orange")
plt.plot(fpr6,tpr6,label="XG Boost",c="grey")
plt.xlabel("False-Positive Rate (FPR)")
plt.ylabel("True-Positive Rate (TPR)")
plt.legend()
plt.show()

# calculate AUROC scores
auroc_dtc = roc_auc_score(y_test, predict_dtc)
auroc_logreg = roc_auc_score(y_test, predict_logreg)
auroc_random = roc_auc_score(y_test, predict_random)
auroc_ensemble = roc_auc_score(y_test, predict_ensemble)
auroc_knn = roc_auc_score(y_test, predict_knn)
auroc_svm = roc_auc_score(y_test, predict_svm)
auroc_bst = roc_auc_score(y_test, predict_bst)

print("AUROC-Score for Decision Tree: {}".format(auroc_dtc))
print("AUROC-Score for Logistic Regression: {}".format(auroc_logreg))
print("AUROC-Score for Random Forest: {}".format(auroc_random))
print("AUROC-Score for Ensemble: {}".format(auroc_ensemble))
print("AUROC-Score for kNN: {}".format(auroc_knn))
print("AUROC-Score for Support Vector Machine: {}".format(auroc_svm))
print("AUROC-Score for XG Boost: {}".format(auroc_bst))
