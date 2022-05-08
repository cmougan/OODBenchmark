# %%
from pmlb import classification_dataset_names, regression_dataset_names
from benchmark import benchmark_experiment
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
import warnings

warnings.filterwarnings("ignore")

# %%
classification_dataset_names_sample = classification_dataset_names[:100]
# %%

modelitos = [
    LogisticRegression(),
    SVC(),
    GaussianProcessClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
]
for m in modelitos:
    benchmark_experiment(
        datasets=classification_dataset_names_sample, model=m, classification=True
    )
