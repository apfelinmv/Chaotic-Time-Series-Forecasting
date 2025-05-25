import numpy as np
import bottleneck as bn

def target_calculation(alphas, last_index, X):
    target_values = {}
    for alpha in alphas:
        offsets = np.cumsum(alpha[1:][::-1])[::-1]
        idxs = last_index - offsets
        target_values[tuple(alpha)] = X[idxs.astype(int)]
    return target_values

def z_prediction(alphas, z_dict, y_alphas, l, threshold):
    preds = []
    for alpha in alphas:
        t = tuple(alpha)
        Z = z_dict[t]                  
        Z_vec = Z[:, :l]               
        dists = np.linalg.norm(Z_vec - y_alphas[t], axis=1)
        valid = Z[dists < threshold, l]  
        if valid.size > 0:
            preds.append(bn.nanmean(valid))
    return bn.nanmean(preds) if preds else np.nan

class CyclePredictor:
    def __init__(self, l=4, k=10):
        self.l = l
        self.k = k
        self.alphas = None
        self.Z = {}
        self.X = None

    def fit(self, X: np.ndarray):
        self.X = X.copy()
        n = len(X)
        idxs = np.indices([self.k]*self.l).reshape(self.l, -1).T + 1
        self.alphas = np.hstack([np.ones((idxs.shape[0],1),int), idxs])
        for alpha in self.alphas:
            offs = np.cumsum(alpha) - alpha[0]
            rows = np.arange(0, n - offs[-1])[:,None] + offs[None,:]
            self.Z[tuple(alpha)] = self.X[rows]
        return self

    def predict(self, steps: int, threshold: float):
        preds = []
        for i in range(steps):
            y_alphas = target_calculation(self.alphas, len(self.X), self.X)
            p = z_prediction(self.alphas, self.Z, y_alphas, self.l, threshold)
            self.X = np.append(self.X, p)
            preds.append(p)
            print(f"Step {i+1}/{steps}: {p:.6f}")
        return preds

