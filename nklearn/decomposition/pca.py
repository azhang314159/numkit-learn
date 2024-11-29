import numpy as np

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.mean_ = None

    def fit(self, X):
        X = np.array(X)
        if not self.n_components:
            self.n_components = X.shape[1]

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        cov = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues_sorted = eigenvalues[sorted_idx]
        eigenvectors_sorted = eigenvectors[:, sorted_idx]

        self.explained_variance_ = eigenvalues_sorted[:self.n_components]
        self.components_ = eigenvectors_sorted[:, :self.n_components].T

        return self

    def transform(self, X):
        X = np.array(X)
        X_centered = X - self.mean_
        X_transformed = X_centered @ self.components_.T
        return X_transformed

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        return X_transformed @ self.components_ + self.mean_