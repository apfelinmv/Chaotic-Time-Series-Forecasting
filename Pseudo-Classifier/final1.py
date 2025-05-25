import numpy as np
from methods_final1 import (
    target_calculation,
    classifier_prediction,
    partial_model,
    train_combiner
)


class ChaoticPrediction:
    def __init__(self, l=4, k=10, combiner_points=None, threshold=1.0):
        self.l = l
        self.k = k
        self.threshold = threshold
        self.combiner_points = combiner_points or k

        self.z = {}
        self.alphas = None
        self.X = None
        self.regressor = None

    def fit(self, X: np.ndarray):
        self.X = X.copy()
        last_index = len(self.X)

        idxs = np.indices([self.k] * self.l).reshape(self.l, -1).T + 1
        self.alphas = np.hstack([np.ones((idxs.shape[0], 1), int), idxs])

        for alpha in self.alphas:
            offsets = np.cumsum(alpha) - alpha[0]
            rows = np.arange(0, last_index - offsets[-1])[:, None] + offsets[None, :]
            self.z[tuple(alpha)] = self.X[rows]

        print(f"Training {len(self.alphas)} trees...")
        partial_model(self.z, self.combiner_points)
        print("All trees trained.")

        print(f"Training combiner regressor on last {self.combiner_points} points...")
        self.regressor = train_combiner(
            self.alphas,
            self.X,
            self.combiner_points
        )
        print("Combiner regressor trained.")
        return self


    def predict(self, steps: int, true_values: np.ndarray = None):
        preds = []
        labels = [] if true_values is not None else None

        for i in range(steps):
            y_alphas = target_calculation(self.alphas, len(self.X), self.X)
            vec = classifier_prediction(y_alphas).reshape(-1)

            if np.any(np.isnan(vec)):
                valid = vec[~np.isnan(vec)]
                if valid.size == 0:
                    vec[:] = 0.0
                else:
                    nan_idx = np.where(np.isnan(vec))[0]
                    vec[nan_idx] = np.random.choice(valid, size=len(nan_idx))

            p = self.regressor.predict(vec.reshape(1, -1))[0]
            preds.append(p)

            is_unreliable = False
            if true_values is not None:
                err = abs(p - true_values[i])
                label = int(err > self.threshold)
                labels.append(label)
                is_unreliable = (label == 1)

            new_val = np.nan if is_unreliable else p
            self.X = np.append(self.X, new_val)
            last = len(self.X) - 1
            for alpha in self.alphas:
                offs = np.cumsum(alpha) - alpha[0]
                idxs_last = last - offs
                row = self.X[idxs_last]
                self.z[tuple(alpha)] = np.vstack([self.z[tuple(alpha)], row])

        preds = np.array(preds)
        if labels is not None:
            labels = np.array(labels)
        return preds, labels

