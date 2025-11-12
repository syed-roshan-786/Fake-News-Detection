import pandas as pd
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")
print("Fake:", len(fake))
print("Real:", len(true))
