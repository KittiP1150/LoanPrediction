import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

#pull data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Data_Link = os.path.join(BASE_DIR, "data", "loan_data_set.csv")
data = pd.read_csv(Data_Link)
data = data.drop(['Loan_ID'], axis = 1)
print(data.head())
print(data.info())
print(data.describe().T)