from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

iris_dataset=load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"])
kn = KNeighborsClassifier()
kn.fit(X_train, y_train)
prediction = kn.predict(X_test)
print(f"ACCURACY: {kn.score(X_test, y_test)}")
target_names = iris_dataset.target_names
for pred,actual in zip(prediction,y_test):
    print(f"Prediction is {target_names[pred]} Actual is {target_names[actual]}")