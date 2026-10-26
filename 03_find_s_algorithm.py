# Practice 3: FIND-S algorithm
import pandas as pd

data = pd.read_csv("EnjoySport.csv")
hypothesis = None

for _, row in data.iterrows():
    x = row.iloc[:-1].tolist()
    y = row.iloc[-1]
    if str(y).strip().lower() == "yes":
        if hypothesis is None:
            hypothesis = x.copy()
        else:
            for i in range(len(hypothesis)):
                if hypothesis[i] != x[i]:
                    hypothesis[i] = "?"

print("Most Specific Hypothesis:")
print(hypothesis)
