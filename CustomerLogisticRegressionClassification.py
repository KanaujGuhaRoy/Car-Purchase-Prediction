import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importing the dataset
dataset = pd.read_csv('CustomerData.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
print(X) 
print(y)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                        test_size = 0.25, random_state = 0)
# k=X_test
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
 
# Predicting the Test set results
y_pred = classifier.predict(X_test)
#------------------------------------------
from sklearn.metrics import confusion_matrix
print("Confusion Matrix:",confusion_matrix(y_test, y_pred))
from sklearn.metrics import accuracy_score
print("Accuracy: ",accuracy_score(y_test, y_pred))

# New customer Test prediction with age=23 and salary=19000
k=sc.transform([[23,19000]])
print("New customer class:",classifier.predict(k))

