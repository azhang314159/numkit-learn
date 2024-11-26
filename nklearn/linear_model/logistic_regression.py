import numpy as np

class LogisticRegression:
    def __init__(self, fit_intercept=True, learning_rate=0.01, num_iterations=1000):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        beta = np.zeros(X.shape[1])
        for i in range(self.num_iterations):
            gradient = (1 / X.shape[0]) * X.T @ (self._sigmoid(X @ beta) - y)
            beta -= self.learning_rate * gradient
        
        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0
            self.coef_ = beta
        
        return self
    
    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
    
    def predict_proba(self, X):
        X = np.array(X)

        if self.fit_intercept:
            z = self.intercept_ + X @ self.coef_
        else:
            z = X @ self.coef_
        return self._sigmoid(z)

    def score(self, X, y):
        y = np.array(y)
        y_pred = self.predict(X)
        acc = np.mean(y_pred == y)
        return acc