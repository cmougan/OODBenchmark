# %%
import pandas as pd
import os

files = os.listdir("results")
files
# %%

gb = pd.read_csv("results/GradientBoostingRegressor_explain.csv", index_col=0)
# %%
gb.mean()
# %%
