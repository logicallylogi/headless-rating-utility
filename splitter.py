import pandas as pd

extract = input("What column should be extracted?\n> ")
df = pd.read_csv('input.csv', usecols = [extract], low_memory = True)

df.to_csv("input", index=False, header=False)