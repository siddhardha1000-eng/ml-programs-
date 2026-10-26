# Practice 2: Loading, preprocessing and analyzing a dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATASET = "student.csv"  # Change this to your dataset filename.

df = pd.read_csv(DATASET)
print("First rows:\n", df.head())
print("\nShape:", df.shape)
df.info()
print("\nDescriptive statistics:\n", df.describe(include="all"))
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# Fill missing age values if the column exists; otherwise drop rows with missing values.
if "age" in df.columns:
    df["age"] = df["age"].fillna(df["age"].median())
else:
    df = df.dropna()
df = df.drop_duplicates()

if "sex" in df.columns:
    df["sex"] = LabelEncoder().fit_transform(df["sex"].astype(str))
if "embarked" in df.columns:
    df = pd.get_dummies(df, columns=["embarked"], drop_first=True)

print("\nPreprocessed dataset:\n", df.head())
# Example split: set TARGET_COLUMN to the target column in your dataset.
TARGET_COLUMN = None
if TARGET_COLUMN and TARGET_COLUMN in df.columns:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Training rows:", len(X_train), "Testing rows:", len(X_test))
