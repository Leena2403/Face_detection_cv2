# -*- coding: utf-8 -*-
"""pandas.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11s9IC6Y-bLxfu8s2YtnJakrwBA2LjTEu
"""

import pandas as pd

# 1D data is a series. 2D Data is DataFrame in pandas.
import seaborn as sns

df = sns.load_dataset("iris")
type(df)

df.head

df.columns

df.index

names = ["Leena","Ravi","Mohit"]
ages = [21,34,45]

d = {
    "names" : names,
    "age": ages
    }
d

pd.DataFrame(d)

import numpy as np

data = np.random.randint(10,20,(10,5))

pd.DataFrame(data)

marks = pd.DataFrame(data, columns=["p","c","m","e","ss"])

marks

marks["bio"] = 0

marks

df.values

df.describe()

df.sort_index(axis=1)

marks.sort_index(axis=1)

marks.sort_values(["e"])

marks

# iloc is used for indexing
marks.iloc[[0,2,6],[0,2,3]]

