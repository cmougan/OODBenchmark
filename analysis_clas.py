# %%
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import rcParams

plt.rcParams.update({"font.size": 22})
# sns.set_style(style="whitegrid")
# plt.style.use('seaborn-whitegrid')
rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8

files = os.listdir("results")
files
# %%
df = pd.DataFrame()
for file in files:
    if "clas" in file:
        aux = pd.read_csv(os.path.join("results", file), index_col=0)
        df = df.append(aux)

df = df.reset_index()
split = df["index"].str.split("_column_", expand=True)
df = df.assign(data=split[0], column=split[1])

df = df.drop(columns="index")

# %%
df.groupby("model").agg(["mean", "std"]).to_csv("results/clas_results.csv")

# %%
plt.figure(figsize=(15, 9))
plt.title(
    "Out-of-distribution model performance on {} datasets for {} columns".format(
        df["data"].nunique(), df.shape[0]
    )
)
sns.boxplot(data=df, x="model", y="oodError", notch=True)
plt.grid(True, axis="y")
plt.ylabel("AUC")
plt.xlabel("")
plt.savefig("images/classOODperf.png")
# %%
plt.figure(figsize=(15, 9))
plt.title(
    "Out-of-distribution error classification on {} datasets for {} columns".format(
        df["data"].nunique(), df.shape[0]
    )
)
sns.boxplot(data=df, x="model", y="oodPerformance", notch=True)
plt.grid(True, axis="y")
plt.ylabel("OOD Error")
plt.xlabel("")
plt.savefig("images/classOODerror.png")

# %%
