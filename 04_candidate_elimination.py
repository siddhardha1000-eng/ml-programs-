# Practice 4: Candidate Elimination algorithm
# This is a straightforward classroom demonstration of S and G boundary updates.
import pandas as pd

data = pd.read_csv("EnjoySport.csv")
concepts = data.iloc[:, :-1].values.tolist()
target = data.iloc[:, -1].astype(str).str.lower().tolist()

if not concepts:
    raise ValueError("EnjoySport.csv has no training examples.")

num_attributes = len(concepts[0])
specific = concepts[0].copy()
general = [["?" for _ in range(num_attributes)] for _ in range(num_attributes)]

for i, example in enumerate(concepts):
    if target[i] == "yes":
        for j in range(num_attributes):
            if example[j] != specific[j]:
                specific[j] = "?"
                general[j][j] = "?"
    else:
        for j in range(num_attributes):
            if example[j] != specific[j]:
                general[j][j] = specific[j]
            else:
                general[j][j] = "?"

general = [g for g in general if g != ["?"] * num_attributes]
print("Specific Boundary:")
print(specific)
print("\nGeneral Boundary:")
for item in general:
    print(item)
