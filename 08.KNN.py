from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris_datasets = load_iris()
X_train,X_test,Y_train,Y_test = train_test_split(iris_datasets["data"],iris_datasets["target"])
kn = KNeighborsClassifier()
kn.fit(X_train,Y_train)
prediction = kn.predict(X_test)
print(f"Accuracy is {kn.score(X_test,Y_test)}\n\n")
target_names = iris_datasets.target_names
for pred,actual in zip(prediction,Y_test):
    print(f"Prediction: {target_names[pred]}\n\nActual:{target_names[actual]}\n\n")