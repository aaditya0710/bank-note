from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data.csv")
print(df.head())

x = df[['variance', 'skewness', 'curtosis', 'entropy']]
y = df['class']

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)

rfc = RandomForestClassifier()
rfc.fit(xtrain,ytrain)

print("accuracy score ",accuracy_score(ytest,rfc.predict(xtest)))

joblib.dump(rfc,"rfc_model.pkl")
print("saved model!!")