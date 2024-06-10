from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels import robust
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,MinMaxScaler
from sklearn.impute import KNNImputer
from scipy.stats import gaussian_kde
import seaborn as sns

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

# Historgram of target
plt.figure(figsize=(8,6))
plt.title("histogram TARGET")
plt.hist(mushroom["poisonous"],bins=2,color="#1ACC94")
plt.ylabel("frequency of bins")
plt.xlabel("target (binary)")
plt.show()

# Historgram "veil-type"
plt.figure(figsize=(8,6))
plt.title("histogram veil-type")
plt.hist(mushroom["veil-type"],bins=2,color="#1ACC94")
plt.ylabel("frequency of bins")
plt.xlabel("veil-type")
plt.show()

# Historgram of odor
plt.figure(figsize=(8,6))
plt.title("histogram odor")
plt.hist(mushroom["odor"],bins=9,color="#1ACC94")
plt.ylabel("frequency of bins")
plt.xlabel("odor")
plt.show()

# ENCODING
# Label-Encoding of all values
label_encoders = {}
for column in mushroom.columns:
    encoder = LabelEncoder()
    mushroom[column] = encoder.fit_transform(mushroom[column])

# Normalizing all values
scaler = MinMaxScaler()

mushroom_scaled = scaler.fit_transform(mushroom)

# Convert the numpy arrays to DataFrames
mushroom_scaled = pd.DataFrame(mushroom_scaled, columns=mushroom.columns)

mushroom = mushroom_scaled

#Adopting values: dropping "veil-type" since it only contains one characteristic
to_drop = ["veil-type"]
mushroom.drop(to_drop, axis=1, inplace=True)

# calculating new correlations
corr_matrix_second = mushroom.corr()
corr_matrix_second

# Filter out correlations where both features are the same (correlation equals 1)
mask = np.eye(len(corr_matrix_second), dtype=bool)
filtered_corr2 = corr_matrix_second.mask(mask)

# Find correlations greater than or equal to 0.5
high_corr2 = filtered_corr2[filtered_corr2 >= 0.5].stack().reset_index()

# Find correlations smaller than or equal to -0.5
high_corr3 = filtered_corr2[filtered_corr2 <= -0.5].stack().reset_index()

print(high_corr2)
print(high_corr3)

# Visualistion of Pearson matrix
def display_correlation(mushroom):
    r = mushroom.corr(method="pearson")
    mask_threshold = ((r > 0.5) | (r < -0.5)) & (r != 1)
    filtered_r = r.where(mask_threshold)
    mask = np.triu(np.ones_like(filtered_r, dtype=bool))
    combined_mask = mask | ~mask_threshold
    plt.figure(figsize=(15, 10))
    heatmap = sns.heatmap(filtered_r, vmin=-1, vmax=1, annot=True, cmap="coolwarm", cbar_kws={'label': 'Correlation Coefficient'}, mask=combined_mask)
    plt.title("Pearson Correlation Coefficient (filtered)")
    plt.show()
    return filtered_r

r_simple = display_correlation(mushroom)


