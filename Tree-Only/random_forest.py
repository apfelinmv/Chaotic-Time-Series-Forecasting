import numpy as np
import bottleneck as bn
from sklearn.tree import DecisionTreeRegressor
from concurrent.futures import ThreadPoolExecutor

trees = dict()

def target_calculation(alphas, last_index, X):
    target_values = {}
    for alpha in alphas:
        idxs = last_index - np.cumsum(np.array(alpha[1:])[::-1])[::-1]
        target_values[tuple(alpha)] = X[idxs.astype(int)]
    return target_values

def random_forest_prediction(alphas, z, y_alphas, l, threshold, steps):
    def process_alpha(alpha):
        t = tuple(alpha)
        if steps == 0:
            mat = z[t]
            X_train, y_train = mat[:, :-1], mat[:, -1]
            model = DecisionTreeRegressor(min_samples_leaf=1, random_state=42, max_depth=365)
            model.fit(X_train, y_train)
            trees[t] = model

        return trees[t].predict(y_alphas[t].reshape(1, -1))[0]

    with ThreadPoolExecutor() as ex:
        preds = list(ex.map(process_alpha, alphas))

    return bn.nanmean(preds)

class ChaoticPrediction:
    def __init__(self, l=4, k=10):
        self.l = l
        self.k = k
        self.z = {}
        self.alphas = None
        self.X = None



    def fit(self, X: np.ndarray):
        self.X = X
        last = len(X)
        idxs = np.indices([self.k] * self.l).reshape(self.l, -1).T + 1
        self.alphas = np.hstack([np.ones((idxs.shape[0], 1), int), idxs])
        for alpha in self.alphas:
            offsets = np.cumsum(alpha) - alpha[0]
            rows = np.arange(0, last - offsets[-1])[:, None] + offsets
            self.z[tuple(alpha)] = X[rows]
        return self

    def predict(self, steps, threshold):
        preds = []
        for i in range(steps):
            y_alphas = target_calculation(self.alphas, len(self.X), self.X)
            next_val = random_forest_prediction(
                self.alphas, self.z, y_alphas, self.l, threshold, i
            )
            self.X = np.append(self.X, next_val)
            preds.append(next_val)
            print(f"step {i}: {next_val}")
        return preds
