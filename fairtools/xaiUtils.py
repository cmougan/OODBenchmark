from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import shap


class ShapEstimator(BaseEstimator, ClassifierMixin):
    """
    A ShapValues estimator based on tree explainer.
    Returns the explanations of the data provided self.predict(X)
    """

    def __init__(self, model, explainer):
        self.model = model
        self.explainer = explainer

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        self.model.fit(self.X_, self.y_)
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        explainer = shap.Explainer(self.model)
        shap_values = explainer(X).values

        return shap_values


"""
## Test ##
import xgboost
X, y = shap.datasets.boston()
y_pred = cross_val_predict(xgboost.XGBRegressor(), X, y, cv=3)

"""
