import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("C:\\Users\\mz ahmed\\Desktop\\udemy\\DTR_Data 1\\Petrol_Consumption.csv")
indepX=dataset.drop("Petrol_Consumption",axis=1)
depY=dataset['Petrol_Consumption']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(indepX,depY,test_size=0.2,random_state=0)
from sklearn.tree import DecisionTreeRegressor
DTR=DecisionTreeRegressor(random_state=0)
DTR.fit(X_train,y_train)
y_pred=DTR.predict(X_test)
df=pd.DataFrame({'Actual':y_test,'predicted':y_pred})
print(df)


