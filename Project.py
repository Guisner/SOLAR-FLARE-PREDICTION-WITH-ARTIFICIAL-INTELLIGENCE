import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

#information of data set
data = pd.read_csv('flaredataall.data', sep=" ")
print(data.head())
print(data.shape)
print("--")
#Detection of missing values (No missing values)
print(data.isnull().sum())
print("--")
#Detection of duplicate data
print(data.duplicated().sum())
print(data.info())

#Converting Cclass as binaries (1,0)
for i in range(2, 10):
    data["Cclass24"] = data["Cclass24"].replace([i], 1)

X = data.drop(["Cclass24"], axis=1)
y = data["Cclass24"]

#Converting object data to numerical data (dummy variables)
X = pd.concat([X, pd.get_dummies(X.select_dtypes(include='object'))], axis=1)
X = X.drop(['Zurich', 'LargestSpot', 'SunspotDistribution'], axis=1)
print(X.head())

#Splitting data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Pie chart for original data
labels=["Zero","One"]
plt.pie(data["Cclass24"].value_counts() , labels =labels ,autopct='%.02f' );
plt.show()

#SMOTE
smote = SMOTE(random_state = 2)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train.ravel())

#Pie chart for SMOTE
test = [len(y_train_res[y_train_res == 0]), len(y_train_res[y_train_res == 1])]
labels=["Zero","One"]
plt.pie(test, labels =labels ,autopct='%.02f' );
plt.show()

#Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=120, max_features="auto", random_state=42)
rf_model.fit(X_train_res, y_train_res)
predictions = rf_model.predict(X_test)

#Importance of features
print(classification_report(y_test, predictions))
plt.figure(figsize=(16, 9))
ranking = rf_model.feature_importances_
features = np.argsort(ranking)[::-1][:30]
columns = X.columns
plt.title("Feature importances based on Random Forest Classifier", y = 1.00, size = 19)
plt.bar(range(len(features)), ranking[features], color="aqua", align="center")
plt.xticks(range(len(features)), columns[features], rotation=75)
plt.show()
