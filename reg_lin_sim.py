
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Salary_Data.csv")
X=dataset.iloc[:, :-1].values #var independante iloc (fonction de pd) vas recuperer les indices dont on aura besoin (: pour toute et :-1 pour toute sauf la dernière) 
y=dataset.iloc[:, -1].values # -1 pour la dernière


dataset.describe() # mini description statistique d'une BD

# division du dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1.0/3,random_state=0)

#feature scaling ??? pas besoin d'en faire ici en reg_lin_sim

#construction du mdoel
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# nouvelle prediction
y_pred=model.predict(X_test)

#pour la prediction d'une valeur souhaité on fera
model.predict(10)


# visualisation des resultats 
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, model.predict(X_train), color="blue")
plt.title("salaire vs experiences")
plt.xlabel("experience")
plt.ylabel("salaire")
a=plt.show()































