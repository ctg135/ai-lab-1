import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

columns = ['state', 'size', 'MAE_train', 'MAE_test']

df = pd.read_csv(r'my-test.csv')

print(' 📝 Статисткиа')
print(df.describe())

print(' 🟢 Лучшие MAE_train')
print(df.sort_values(by = ['MAE_train'], ascending=True).head(10))

print(' 🟢 Лучшие MAE_test')
print(df.sort_values(by = ['MAE_test'], ascending=True).head(10))

print(' 🔴 Худшие MAE_train')
print(df.sort_values(by = ['MAE_train'], ascending=False).head(10))

print(' 🔴 Худшие MAE_test')
print(df.sort_values(by = ['MAE_test'], ascending=False).head(10))