import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class P95Capper(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.caps_ = np.percentile(X, 95, axis=0)
        return self

    def transform(self, X):
        # Model lama pakai col_idx, bukan caps_
        if hasattr(self, 'caps_'):
            return np.clip(X, None, self.caps_)
        elif hasattr(self, 'col_idx'):
            # Tidak ada caps tersimpan, skip capping
            return X
        return X