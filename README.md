# LoanPrediction
#Hi ![](https://user-images.githubusercontent.com/18350557/176309783-0785949b-9127-417c-8b55-ab5a4333674e.gif)
This repository explores the trade-offs between different machine learning classifiers for loan approval decisions, focusing on predictive performance, error behavior, and computational efficiency. The project emphasizes application-level considerations rather than accuracy alone.

## Dataset 
The dataset used in this project is obtained from Kaggle:

Loan Prediction Dataset  
Author: Amit Parjapat  
Source: https://www.kaggle.com/datasets/ninzaami/loan-predication  

The dataset is used strictly for educational and analytical purposes under its original license.

## Library Used
- scikit-learn
- pandas
- matplotlib
- NumPy

## Models Evaluated
- Logistic Regression (L1 regularization)
- Decision Tree
- Support Vector Classifier (RBF kernel)

## Project Structure
'''
loanpredict/
├── data/
│ └── loan_data_set.csv
├── model/
│ ├── preprocess.py
│ ├── logistic_regression.py
│ ├── decision_tree.py
│ └── svc.py
├── tuning_dt.py
├── tuning_lr.py
└── run_compare.py
'''


## Note
This project is intended for educational and analytical purposes and does not represent a production-ready credit scoring system.

## License
This project is licensed under the MIT License.
