import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_excel('file.xlsx')
fig, axes = plt.subplots(rows=4, cols=4, figsize=(12,12))
data_cols = df.columns[3:19]

# iterate over pairs of data columns and plot axes
for data_col, ax in zip(data_cols, axes.ravel()):
    ax.plot(df[data_col])


plt.figure(figsize=(10,10))
plt.style.use('seaborn')
plt.scatter(x,y,marker="*",s=100,edgecolors="black",c="yellow")
plt.title("Excel sheet to Scatter Plot")
plt.show()