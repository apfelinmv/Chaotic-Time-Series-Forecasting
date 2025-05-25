import numpy as np
from methods_combined import (
    target_calculation,
    classifier_prediction,
    partial_model,
    train_combiner,
    train_classifier
)


class ChaoticPrediction:
    def __init__(self, l=4, k=10, combiner_points=None, threshold=0.1):
        self.l = l
        self.k = k
        self.threshold = threshold
        self.combiner_points = combiner_points or k

        self.z = {}
        self.alphas = None
        self.X = None
        self.regressor = None
        self.classifier = None

    def fit(self, X: np.ndarray):
        self.X = X.copy()
        N = len(self.X)

        idxs = np.indices([self.k] * self.l).reshape(self.l, -1).T + 1
        raw_alphas = np.hstack([np.ones((idxs.shape[0], 1), int), idxs])
        self.alphas = [tuple(alpha) for alpha in raw_alphas]

        for alpha in self.alphas:
            offsets = np.cumsum(alpha) - alpha[0]
            rows = np.arange(0, N - offsets[-1])[:, None] + offsets[None, :]
            self.z[alpha] = self.X[rows]

        print(f"Training {len(self.alphas)} base trees...")
        partial_model(self.z, self.combiner_points)
        print("Base trees trained.")

        print(f"Training classifier on last {self.combiner_points} points...")
        self.classifier = train_classifier(
            self.alphas,
            self.z,
            self.X,
            self.combiner_points,
            self.threshold
        )
        print("Classifier trained.")

        print(f"Training combiner regressor on last {self.combiner_points} points...")
        self.regressor = train_combiner(
            self.alphas,
            self.X,
            self.combiner_points
        )
        print("Combiner regressor trained.")

        return self

    def predict(self, steps: int):
        preds = []
        labels = []

        for _ in range(steps):
            y_alphas = target_calculation(self.alphas, len(self.X), self.X)
            vec = classifier_prediction(y_alphas).reshape(-1)

            if np.any(np.isnan(vec)):
                valid = vec[~np.isnan(vec)]
                if valid.size == 0:
                    vec[:] = 0.0
                else:
                    vec[np.isnan(vec)] = np.random.choice(valid, size=np.isnan(vec).sum())

            label = int(self.classifier.predict(vec.reshape(1, -1))[0])
            labels.append(label)

            p = self.regressor.predict(vec.reshape(1, -1))[0]
            preds.append(p)

            new_val = np.nan if label == 1 else p
            self.X = np.append(self.X, new_val)
            last = len(self.X) - 1
            for alpha in self.alphas:
                offsets = np.cumsum(alpha) - alpha[0]
                idxs_last = last - offsets
                self.z[alpha] = np.vstack([self.z[alpha], self.X[idxs_last]])

        return np.array(preds), np.array(labels)