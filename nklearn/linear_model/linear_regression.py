import numpy as np

class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        theta = np.linalg.inv(X.T @ X) @ X.T @ y
        if self.fit_intercept:
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0
            self.coef_ = theta

        return self

    def predict(self, X):
        X = np.array(X)

        if self.fit_intercept:
            y_pred = self.intercept_ + X @ self.coef_
        else:
            y_pred = X @ self.coef_
        return y_pred

    def score(self, X, y):
        y = np.array(y)
        y_pred = self.predict(X)
        sst = np.sum((y - np.mean(y)) ** 2)
        ssr = np.sum((y - y_pred) ** 2)
        return 1 - ssr / sst