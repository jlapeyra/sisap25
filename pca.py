from abc import ABC, abstractmethod
import faiss
import numpy as np
from sklearn import decomposition


class PCA(ABC):
    name:str
    d_in:int
    d_out:int

    @abstractmethod
    def __init__(self, d_in, d_out):
        pass

    @abstractmethod
    def fit(self, X) -> None:
        """
        Fit the PCA model to the data X.
        :param X: Input data of shape (n_samples, n_features).
        """
        pass

    @abstractmethod
    def transform(self, X) -> np.ndarray:
        """
        Transform the data X using the fitted PCA model.
        :param X: Input data of shape (n_samples, n_features).
        :return: Transformed data of shape (n_samples, n_components).
        """
        pass

    @abstractmethod
    def fit_transform(self, X) -> np.ndarray:
        """
        Fit the PCA model to the data X and then transform it.
        :param X: Input data of shape (n_samples, n_features).
        :return: Transformed data of shape (n_samples, n_components).
        """
        pass

    # def __init__(self, version, given_d, target_d):
    #     if version == 'faiss':
    #         self.__class__ = PCA_faiss
    #     elif version == 'sklearn':
    #         self.__class__ = PCA_sklearn


class PCA_sklearn(PCA):
    name = 'sklearn'

    def __init__(self, d_in, d_out):
        self.d_in = d_in
        self.d_out = d_out
        self.pca = decomposition.PCA(n_components=d_out, svd_solver='randomized', random_state=42)

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.pca.fit(X)

    def transform(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        return self.pca.transform(X)

    def fit_transform(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        return self.pca.fit_transform(X)

class PCA_faiss(PCA):
    name = 'faiss'

    def __init__(self, d_in, d_out):
        self.d_in = d_in
        self.d_out = d_out
        self.pca = faiss.PCAMatrix(d_in, d_out)

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.pca.train(X)

    def transform(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        return self.pca.apply_py(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)