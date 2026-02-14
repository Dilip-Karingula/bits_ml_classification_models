from click import option
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
import joblib

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_log_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef,roc_auc_score


st.title("Machine Learning Assignment - 2")
st.title("Classification Models")

st.markdown(
    """"""
)

##File uploader
uploaded_file = st.file_uploader("Please upload excel file with Data", type=["csv"])

if uploaded_file is not None:

    ds = pd.read_csv(uploaded_file)
    
    #st.write("""File Data:""", ds.shape)
    #st.write(ds)
    
    feature_names = ['tournament_name','team1','team2','venue','innings1_team','innings1_runs','innings1_wkts','innings1_overs','innings2_team','innings2_runs','innings2_wkts','innings2_overs']
    x = ds.loc[:,feature_names].values
    y = ds.loc[:,['winner']].values

    ##Encode Data
    # Identify categorical features for OneHotEncoding
    categorical_features_indices = [0, 1, 2, 3, 4, 8] # Indices for 'tournament_name','team1','team2','venue','innings1_team','innings2_team'

    # Apply OneHotEncoder to the specified categorical features
    ohe = ColumnTransformer([("one_hot_encoder", OneHotEncoder(), categorical_features_indices)], remainder='passthrough')
    x = ohe.fit_transform(x)

    # Encode the target variable 'winner'
    le = LabelEncoder()
    y = le.fit_transform(y.ravel())
    #st.write("""Shape of x after OneHotEncoding:""", x.shape)
    #st.write("""Shape of y after LabelEncoding:""", y.shape)

    #Data Clean up
    # Identify rows with NaNs in x_encoded or y_encoded
    nan_mask = np.isnan(x.toarray()).any(axis=1) | np.isnan(y).any()
   # st.write("""Number of samples before dropping NaNs:""", x.shape[0])

    x = x[~nan_mask]
    y = y[~nan_mask]

    class_counts = pd.Series(y).value_counts()
    single_member_classes = class_counts[class_counts < 2].index

    valid_samples_mask = ~pd.Series(y).isin(single_member_classes)

    x = x[valid_samples_mask.values]
    y = y[valid_samples_mask.values]

    
    if x is not None:
        st.success("File uploaded successfully!")

        ##Options for classification models

        option = st.selectbox(
        "Select Classification model",
        ("Select","Logistic regression", "Decision Tree", 
        "kNN","Naive Bayes","Random Forest (Ensemble)","XGBoost (Ensemble)"))
    
        if option is not None and option != "Select":
            #st.warning("Selected:"+ option)
    
            #st.write("""Number of samples after dropping NaNs and single-member classes:""", x.shape[0])
            
            
            #Perform prediction based on selected model
            y_pred_lr = None
            
            if option == "Logistic regression":
                #Load models
                logisticRegressionModel = joblib.load("model/pkl/logisticRegression.pkl") 
                y_pred_lr = logisticRegressionModel.predict(x)


            #Metrics
            if y_pred_lr is not None:
                accuracy = accuracy_score(y, y_pred_lr)
                precision = precision_score(y, y_pred_lr, average='weighted', zero_division=0)
                recall = recall_score(y, y_pred_lr, average='weighted', zero_division=0)
                f1 = f1_score(y, y_pred_lr, average='weighted', zero_division=0)
                mcc = matthews_corrcoef(y, y_pred_lr)

                y_proba = logisticRegressionModel.predict_proba(x)
                auc = roc_auc_score(y, y_proba, multi_class='ovr', average='weighted', labels=logisticRegressionModel.classes_)

                #st.write(f"Accuracy: {accuracy:.4f}")
                #st.write(f"Precision: {precision:.4f}")
                #st.write(f"Recall: {recall:.4f}")
                #st.write(f"F1-Score: {f1:.4f}")
                #st.write(f"MCC: {mcc:.4f}")
                #st.write(f"AUC (Weighted): {auc:.4f}")

                # Display Accuracy and AUC in table format
                st.write("Evaluation Metrics for " + option)

                metrics_df = pd.DataFrame({
                    'Metric Type': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'AUC (Weighted)'],
                    'Score': [accuracy, precision, recall, f1, mcc, auc]
                })
                st.table(metrics_df)
    else:
        st.info("Model predictions not yet implemented for this option.")
#else:
    #st.warning("Please upload a file to proceed")

