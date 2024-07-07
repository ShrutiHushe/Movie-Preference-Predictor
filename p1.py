#import lib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

#load the data

data = pd.read_csv("am_march24.csv")
print(data)

#check for null data
print(data.isnull().sum())

#check for dupliacted data
print(data.duplicated().sum())

#feature and target
feature = data[["age"]]
target = data["movie"]

#train test
x_train, x_test, y_train, y_test = train_test_split(feature.values, target, stratify=target)

#model
model = LogisticRegression()
model.fit(x_train, y_train)

#performance
cm = confusion_matrix(y_test, model.predict(x_test))
print(cm)

cr = classification_report(y_test, model.predict(x_test))
print(cr) 

#prediction
age = float(input("Enter age "))
m1 = model.predict([[age]])
print(m1)

#internal working
pp = model.predict_proba([[age]])
print(pp)
ppp = pp.ravel().tolist()   #2d into id
print(ppp)

ddlj = round(ppp[0] * 100, 2)
harrypotter= round(ppp[1] * 100, 2)
moneyheist = round(ppp[2] * 100, 2)
silsila = round(ppp[3] * 100, 2)

print("ddlj ", ddlj, "%")
print("harrypotter ", harrypotter, "%")
print("moneyheist ", moneyheist, "%")
print("silsila ", silsila, "%")