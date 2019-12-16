import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

df=pd.read_csv("insurance_data.csv")
print(df.head())

plt.scatter(df.age,df.bought_insurance,marker="+",color="red")
plt.show()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df[["age"]],df.bought_insurance, test_size = 0.5, random_state = 0)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(x_test)
print(y_pred)
print(model.score(x_test,y_test))
print(model.predict_proba(x_test))
#print(confusion_matrix(y_test, y_pred))
