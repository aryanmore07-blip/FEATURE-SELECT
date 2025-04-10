
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif, RFE
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Feature SelectX", layout="wide")
st.title("🎯 Feature SelectX – Smart Feature Selection Tool")

uploaded_file = st.file_uploader("📁 Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Data Preview")
    st.dataframe(df.head())

    target = st.selectbox("🎯 Select your target column", df.columns)
    features = df.drop(columns=[target])
    X = features.select_dtypes(include=['int64', 'float64'])  # numerical features only
    y = df[target]

    method = st.selectbox("📌 Choose a feature selection method", 
                          ["Mutual Info", "Chi-Square", "ANOVA F-Test", "RFE"])

    show_plot = st.checkbox("📊 Show visualization", value=True)

    # Compute feature importance
    scores = None
    if method == "Mutual Info":
        scores = pd.Series(mutual_info_classif(X.fillna(0), y), index=X.columns)
    elif method == "Chi-Square":
        scores = pd.Series(chi2(abs(X.fillna(0)), y)[0], index=X.columns)
    elif method == "ANOVA F-Test":
        scores = pd.Series(f_classif(X.fillna(0), y)[0], index=X.columns)
    elif method == "RFE":
        model = LogisticRegression(max_iter=1000)
        selector = RFE(model, n_features_to_select=5)
        selector = selector.fit(X.fillna(0), y)
        scores = pd.Series(selector.ranking_, index=X.columns)
        scores = scores.map(lambda x: 1 if x == 1 else 0)

    score_series = scores.sort_values(ascending=False)

    if show_plot:
        st.subheader(f"📊 Feature Importance ({method})")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=score_series.values, y=score_series.index, ax=ax)
        st.pyplot(fig)

    selected = st.multiselect("✅ Pick features to keep", score_series.index.tolist(), default=score_series.index[:5])
    if st.button("💾 Download Selected Features"):
        selected_df = df[selected + [target]]
        selected_df.to_csv("selected_features.csv", index=False)
        st.success("✅ File ready to download below")
        st.download_button("📥 Download CSV", data=selected_df.to_csv(index=False), file_name="selected_features.csv", mime="text/csv")
