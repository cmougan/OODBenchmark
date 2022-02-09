# %%
from pmlb import fetch_data
from pmlb import classification_dataset_names, regression_dataset_names
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
import re
import traceback
import sys

warnings.filterwarnings("ignore")


# %%
classification_dataset_names
classification_dataset_names_sample = [
    "allhypo",
    "allrep",
]
# %%
# %%
regression_dataset_names_sample = regression_dataset_names[:10]
# %%
benchmark(datasets=regression_dataset_names_sample, model=DecisionTreeRegressor())
## Linear Regression
benchmark(datasets=regression_dataset_names_sample, model=LinearRegression())

# %%
modelitos = [Lasso(),RandomForestRegressor(),DecisionTreeRegressor(),GradientBoostingRegressor(),]
for m in modelitos:
    benchmark(datasets=regression_dataset_names_sample, model=m)

# %%
