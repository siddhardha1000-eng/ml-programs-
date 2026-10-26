# Practice 10: Remove duplicates, cross-validation, bias and variance
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
df = pd.concat([df, df.iloc[:5]], ignore_index=True)

print("Dataset Shape Before Removing Duplicates:", df.shape)
df = df.drop_duplicates()
print("Dataset Shape After Removing Duplicates:", df.shape)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

scores = cross_val_score(model, X, y, cv=5)
print("\nCross Validation Scores:", scores)
print("Average Cross Validation Accuracy:", scores.mean())
print("\nBias: Error due to overly simple assumptions.")
print("Variance: Error due to model sensitivity to training data.")
