import streamlit as st
import pandas as pd

st.title("Machine Learning Assignment - 2 : Classification Models")
st.markdown(
    """Implementation of classification models."""
)

option = st.selectbox(
    "Select Classification model",
    ("Logistic regression", "Decision Tree", 
     "kNN","Naive Bayes","Random Forest (Ensemble)","XGBoost (Ensemble)"))

st.write("Selected:", option)


uploaded_file = st.file_uploader("Choose a file")
ds = pd.read_csv(uploaded_file)
st.write(ds)

feature_names = ['tournament_name','team1','team2','venue','innings1_team','innings1_runs','innings1_wkts','innings1_overs','innings2_team','innings2_runs','innings2_wkts','innings2_overs']
x = ds.loc[:,feature_names].values
y = ds.loc[:,['winner']].values

st.write("""Shape of x after OneHotEncoding:""", x.shape)
st.write("""Shape of y after LabelEncoding:""", y.shape)


