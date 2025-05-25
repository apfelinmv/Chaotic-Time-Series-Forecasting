import time
import numpy as np
import matplotlib.pyplot as plt
from final1 import ChaoticPrediction

def run_combined_prediction(
    series: np.ndarray,
    n_train: int,
    n_test: int,
    l: int = 4,
    k: int = 10,
    combiner_points: int = None,
    threshold: float = 1,
    plot: bool = True
):
    X_train = series[:n_train]
    actual = series[n_train : n_train + n_test]

    if combiner_points is None:
        combiner_points = k

    start = time.time()
    model = ChaoticPrediction(
        l=l, k=k,
        combiner_points=combiner_points,
        threshold=threshold
    )
    model.fit(X_train)
    print(f"Fit time: {time.time() - start:.3f} s")

    start = time.time()
    predictions, labels = model.predict(
        steps=n_test,
        true_values=actual
    )
    print(f"Prediction time: {time.time() - start:.3f} s")

    mae = np.abs((predictions - actual))
    mse = np.abs((predictions - actual)**2)
    mse = mse.mean()
    mae = mae.mean()
    print(f"MAE: {mae:.6f}")
    print(f"MSE: {mse:.6f}")

    # 6) Plot
    if plot:
        idx = np.arange(n_test)
        plt.figure(figsize=(12, 8))
        plt.plot(idx, actual,       label='Actual',    linewidth=1)
        plt.plot(idx, predictions, '--', label='Predicted')
        reliable   = idx[labels == 0]
        unreliable = idx[labels == 1]
        plt.scatter(reliable,   actual[reliable],   s=50, label='Reliable (0)')
        plt.scatter(unreliable, actual[unreliable], s=50, label='Unreliable (1)')
        plt.title('Combined Regime: Actual vs Predicted with Pseudo-Classifier')
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()

    return actual, predictions, labels, mse
