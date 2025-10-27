import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

df = pd.read_csv('data/Customer_churn_csv')
X = df.drop(('CustomerId','Surname','Exited'),axis=1)
y = df['Exited']

X['Geography'] = LableEncoder().fit_transform(X['Geography'])
X['Gender'] = LableEncoder().fit_transform(X['Gender'])

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)
scaler = 