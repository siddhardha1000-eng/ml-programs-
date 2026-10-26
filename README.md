# Machine Learning Lab Programs (B.Tech V Semester, 2026–2027)

This repository separates the programs from the Machine Learning Lab manual into individual Python files.

## Programs
1. `01_basic_python_commands.py` — basic Python, NumPy, Pandas and Matplotlib
2. `02_data_preprocessing.py` — load, analyze and preprocess a CSV dataset
3. `03_find_s_algorithm.py` — FIND-S algorithm (`EnjoySport.csv`)
4. `04_candidate_elimination.py` — Candidate Elimination (`EnjoySport.csv`)
5. `05_binary_classification.py` — Logistic Regression purchase prediction (`Social_Network_Ads.csv`)
6. `06_knn_iris.py` — K-Nearest Neighbours on Iris
7. `07_linear_regression.py` — house-price prediction (`HousePrice.csv`)
8. `08_logistic_regression.py` — Logistic Regression purchase prediction (`Social_Network_Ads.csv`)
9. `09_svm_heart_disease.py` — linear SVM (`heart.csv`)
10. `10_duplicates_bias_variance_cross_validation.py` — duplicate removal and cross-validation
11. `11_one_hot_encoding.py` — One-Hot Encoding
12. `12_categorical_encoding.py` — Label Encoding
13. `13_random_forest.py` — Random Forest on Iris
14. `14_bagging_boosting_sentiment.py` — Bagging and AdaBoost on sentiment data (`twitter_training.csv`)

## Setup
Install Python 3.10+ and run:

```bash
python -m pip install -r requirements.txt
```

Run a program from this folder, for example:

```bash
python 03_find_s_algorithm.py
```

## Dataset notes
- `EnjoySport.csv` and `HousePrice.csv` are included.
- Programs 2, 5, 8, 9 and 14 expect the named CSV datasets. Add your class dataset files to this folder before running those programs.
- For Program 2, update the column names (`age`, `sex`, `embarked`) to match your own dataset.
- For Program 14, the code expects columns named `Tweet` and `Sentiment`. Some Kaggle datasets use different column names; adjust the code to match the downloaded file.
- Program 1's package-install command belongs in the terminal, not inside a Python script.

The code files are separated and organized for GitHub. Check the dataset names and column names before running programs that use external CSV files.
