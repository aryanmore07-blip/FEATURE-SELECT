# -*- coding: utf-8 -*-
"""FEATURE SELECT.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15UeKmddmtaZ7TKjxYQuFssNdv-bSqlWl
"""

import pandas as pd
from google.colab import files

uploaded = files.upload()

df = pd.read_csv(next(iter(uploaded)))
df.head()

import ipywidgets as widgets
from IPython.display import display

target_col = widgets.Dropdown(options=df.columns, description='Target:')
display(target_col)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

selected_target_col = target_col.value

X = df.drop(selected_target_col, axis=1)
y = df[selected_target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

method_selector = widgets.SelectMultiple(
    options=['Chi-Square', 'RFE', 'INFO Gain', 'Anova'],
    description='Methods:',
    rows=5
)
display(method_selector)

import pandas as pd
from sklearn.feature_selection import chi2, RFE, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



def feature_selection(method_selector):

    if method == 'Chi-Square':
        scores, _ = chi2(abs(X_train_scaled), y_train)
        feature_scores = pd.Series(scores, index=X.columns)
    elif method == 'RFE':
        estimator = LogisticRegression(max_iter=1000)
        selector = RFE(estimator, n_features_to_select=5, step=1)
        selector = selector.fit(X_train_scaled, y_train)
        feature_scores = pd.Series(selector.ranking_, index=X.columns)
    elif method == 'INFO Gain':
        scores = mutual_info_classif(X_train_scaled, y_train)
        feature_scores = pd.Series(scores, index=X.columns)
    elif method == 'Anova':
        scores, _ = f_classif(X_train_scaled, y_train)
        feature_scores = pd.Series(scores, index=X.columns)
    else:
        raise ValueError(f"Invalid method: {method}")

    sorted_scores = feature_scores.sort_values(ascending=False if method != 'RFE' else True)  # RFE ranking is reversed


    top_5_features = sorted_scores.head(5) if method != 'RFE' else sorted_scores[sorted_scores == 1].index

    return top_5_features


methods = method_selector.value
for method in methods:
    print(f"\nFeature Selection using {method}:")
    results = feature_selection(method)
    print(results)

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_scores(scores, title="Feature Importance"):
    plt.figure(figsize=(10, 5))
    if isinstance(scores, pd.Index):
        scores = pd.Series(1, index=scores)  # RFE selected features get score 1
    elif isinstance(scores, pd.Series) and title.endswith("using RFE"):
        scores = scores[scores == 1]  # Filter for selected features in RFE
        scores = pd.Series(1, index=scores.index)
    sns.barplot(x=scores.index, y=scores.values)
    plt.title(title)
    plt.xlabel("Feature")
    plt.ylabel("Scores")
    plt.yticks(np.unique(scores.values))
    plt.tight_layout()
    plt.show()

for method in methods:
    print(f"\nFeature Selection using {method}:")
    scores = feature_selection(method)
    plot_scores(scores, title=f"Feature Importance using {method}")

# prompt: on the basis of previous code blocks i just want you to write a code to let the use download the results from whichever feature selection technique they want

import pandas as pd
from google.colab import files
import ipywidgets as widgets
from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, RFE, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ... (your existing code) ...

def download_results(results, method):
    results_df = pd.DataFrame(results)
    results_df.columns = ['Selected Features']
    results_filename = f"feature_selection_results_{method}.csv"
    results_df.to_csv(results_filename, index=False)
    files.download(results_filename)

download_button = widgets.Button(description="Download Selected Features")
display(download_button)

def on_button_clicked(b):
    # Get the currently selected methods
    methods = method_selector.value

for method in methods:
    print(f"\nFeature Selection using {method}:")
    results = feature_selection(method)
    print(results)
    download_results(results, method)


download_button.on_click(on_button_clicked)