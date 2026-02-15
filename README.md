# bits_ml_classification_models
This Project aims to implement various ML classification models

# 1. Problem statement 
      1. Choose any dataset from public repository with minimum 12 features and 500 data instances / samples
      2. Implement the following classification models using the dataset chosen above.
          1. Logistic Regression 
          2. Decision Tree Classifier 
          3. K-Nearest Neighbor Classifier 
          4. Naive Bayes Classifier - Gaussian or Multinomial 
          5. Ensemble Model - Random Forest 
          6. Ensemble Model - XGBoost 
      3. Calculate the following evaluation metrics for each models list above: 
          1. Accuracy 
          2. AUC Score 
          3. Precision 
          4. Recall 
          5. F1 Score 
          6. Matthews Correlation Coefficient (MCC Score) 
      4. Deploy on Streamlit Community Cloud with features
          1. Dataset upload option 
          2. Model selection dropdown
          3. Display of evaluation metrics 
          4. Display Confusion matrix or classification report

# 2. Dataset details
      Description : 
      Source : 
      No. of Features : 
      No. of samples : 
      Features X : 
      Output Feature Y : 
      
# 3. Models implemented
    - Logistic Regression
    - Decision Tree Classifier
    - K-Nearest Neighbor Classifier
    - Naive Bayes Classifier (Gaussian/Multinomial)
    - Random Forest (Ensemble)
    - XGBoost (Ensemble)

# 4. Model Performance Metrics

| Model               | Accuracy | Precision | Recall | F1-Score | MCC Score | AUC Score |
|---------------------|----------|-----------|--------|----------|-----------|-----------|
| Logistic Regression | 0.8808   | 0.8823    | 0.8808 | 0.8775   | 0.8785    | 0.9973    |
| Decision Tree       | 0.9228   | 0.9265    | 0.9228 | 0.9237   | 0.9213    | 0.9607    |
| K-Nearest Neighbor  | 0.2371   | 0.3169    | 0.2371 | 0.2136   | 0.2240    | 0.8832    |
| Naive Bayes         | 0.6160   | 0.6631    | 0.6160 | 0.6054   | 0.6107    | 0.9549    |
| Random Forest       | 0.9456   | 0.9472    | 0.9456 | 0.9454   | 0.9446    | 0.9986    |
| XGBoost             | 0.0839   | 0.0815    | 0.0839 | 0.0827   | 0.0733    | 0.9992    |

# 5. Observations on the performance of each model

| Model               | Observation about model performance                                       |
|---------------------|---------------------------------------------------------------------------|
| Logistic Regression |Good Accuracy, Excellent AUC                                               |
| Decision Tree       |Excellent Accuracy, Excellent AUC but not the best                         |
| K-Nearest Neighbor  |Worst Accuracy, lowest AUC, require fine tuning of model parameters        |
| Naive Bayes         |Moderate Accuracy, Excellent AUC, require fine tuning of model parameters  |
| Random Forest       |Excellent Accuracy, Excellent AUC                                          |
| XGBoost             |Worst Accuracy but Excellent AUC, require fine tuning of model parameters  |

# 6. Streamlit App Functionality
- Dataset upload option.
- Model selection dropdown.
- Display of evaluation metrics.
- Display Confusion matrix

# 7. Key Links
- GitHub Repository: https://github.com/Dilip-Karingula/bits_ml_classification_models
- Live Streamlit App: https://bitsmlassignment2.streamlit.app/