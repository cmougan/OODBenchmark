# %%
import pandas as pd
import os

files = os.listdir("results")
files
# %%
lr = pd.read_csv("results/LogisticRegression_clas.csv", index_col=0)

dt = pd.read_csv("results/DecisionTreeClassifier_clas.csv", index_col=0)
rf = pd.read_csv("results/RandomForestClassifier_clas.csv", index_col=0)
gb = pd.read_csv("results/GradientBoostingClassifier_clas.csv", index_col=0)
# %%
lr.mean()
# %%
lasso.mean()
# %%
dt.mean()
# %%
rf.mean()

# %%
gb.mean()
# %%
files
# %%
