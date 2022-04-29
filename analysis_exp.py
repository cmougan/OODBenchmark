# %%
import pandas as pd
import os
import numpy as np
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
files = os.listdir("results")
files
# %%

gb = pd.read_csv("results/GradientBoostingRegressor_explain.csv", index_col=0)
gb2 = pd.read_csv("results/GradientBoostingRegressor_reg.csv", index_col=0)
# %%
plt.figure()
sns.kdeplot(gb[gb['generalizationError']<1].generalizationError,label='Shap')
sns.kdeplot(gb2.generalizationError,label='Normal')
plt.legend(loc='lower left')
plt.show()

# %%
plt.figure()
sns.kdeplot(gb[np.abs(gb['oodPerformance'])<1].oodPerformance,label='Shap')
sns.kdeplot(gb2.oodPerformance,label='Normal')
plt.legend(loc='lower left')
plt.show()
# %%

# %%
gb2
# %%
