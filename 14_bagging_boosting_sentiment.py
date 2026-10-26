# Practice 14: Bagging and Boosting for sentiment analysis
# Expected CSV columns: Tweet, Sentiment
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("twitter_training.csv")
df = df[["Tweet", "Sentiment"]].dropna()

vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
X = vectorizer.fit_transform(df["Tweet"].astype(str))
y = df["Sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Bagging
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)
bagging.fit(X_train, y_train)
bag_pred = bagging.predict(X_test)
print("Bagging Accuracy:")
print(accuracy_score(y_test, bag_pred))
print("\nBagging Classification Report")
print(classification_report(y_test, bag_pred))

# Boosting (AdaBoost)
boosting = AdaBoostClassifier(n_estimators=100, random_state=42)
boosting.fit(X_train, y_train)
boost_pred = boosting.predict(X_test)
print("\nBoosting Accuracy:")
print(accuracy_score(y_test, boost_pred))
print("\nBoosting Classification Report")
print(classification_report(y_test, boost_pred))
