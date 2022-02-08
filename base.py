# %%
from pmlb import fetch_data
from pmlb import classification_dataset_names, regression_dataset_names
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")


# %%
classification_dataset_names
classification_dataset_names_sample = [
    "allhypo",
    "allrep",
]
# %%

regression_dataset_names_sample = regression_dataset_names[:20]


# %%
# Returns a pandas DataFrame
def benchmark():

    results = defaultdict()
    for i, dataset in enumerate(regression_dataset_names_sample):
        try:
            # Initialise the scaler
            standard_scaler = StandardScaler()

            # Load the dataset and split it
            X, y = fetch_data(dataset, return_X_y=True, local_cache_dir="data/")
            print(X.shape)

            # Scale the dataset
            X = standard_scaler.fit_transform(X)

            # Back to dataframe
            X = pd.DataFrame(X, columns=["Var %d" % (i + 1) for i in range(X.shape[1])])
            data = X.copy()
            data["target"] = y

            # Train test splitting points
            fracc = 0.33
            oneThird = int(data.shape[0] * fracc)
            twoThird = data.shape[0] - int(data.shape[0] * fracc)

            for idx, col in tqdm(enumerate(X.columns), total=len(X.columns)):

                # Sort data on the column
                data = data.sort_values(col).reset_index(drop=True).copy()

                # Train Test Split
                data_sub = data.iloc[:oneThird]
                data_train = data.iloc[oneThird:twoThird]
                data_up = data.iloc[twoThird:]

                X_tot = data.drop(columns="target")
                X_tr = data_train.drop(columns="target")
                X_sub = data_sub.drop(columns="target")
                X_up = data_up.drop(columns="target")

                y_tot = data[["target"]].target.values
                y_tr = data_train[["target"]].target.values
                y_sub = data_sub[["target"]].target.values
                y_up = data_up[["target"]].target.values

                # Fit the estimator
                model = Lasso()

                ## Test predictions
                pred_test = cross_val_predict(
                    estimator=model,
                    X=X_tr,
                    y=y_tr,
                    cv=StratifiedKFold(
                        n_splits=10, shuffle=True, random_state=0
                    ),
                )

                ## Train
                model.fit(X_tr, y_tr)
                pred_train = model.predict(X_tr)

                ## OOD
                X_ood = X_sub.append(X_up)
                y_ood = np.concatenate((y_sub, y_up))
                pred_ood = model.predict(X_ood)

                # Error Calculation
                train_error = mean_squared_error(pred_train, y_tr)
                generalizationError = mean_squared_error(pred_test, y_tr)
                ood_error = mean_squared_error(pred_ood, y_ood) - generalizationError

                # Append Results
                results[dataset] = [train_error, generalizationError, ood_error]
        except:
            print(dataset)
            pass

    df = pd.DataFrame(data=results).T
    df.columns = ["trainError", "testError", "oodError"]
    df.to_csv("results/regression.csv")


# %%
benchmark()
