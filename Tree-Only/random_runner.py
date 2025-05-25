import time
import numpy as np
import matplotlib.pyplot as plt
from random_forest import ChaoticPrediction  # Ваш обновлённый класс


def run_combined_prediction(
    series: np.ndarray,
    n_train: int,
    n_test: int,
    l: int = 4,
    k: int = 10,
    threshold: float = 1,
    plot: bool = True
):
    X_train = series[:n_train]
    actual  = series[n_train : n_train + n_test]

    start = time.time()
    model = ChaoticPrediction(l=l, k=k)
    model.fit(X_train)
    print(f"Fit time: {time.time() - start:.3f} s")

    start = time.time()
    predictions = model.predict(steps=n_test, threshold=threshold)
    print(f"Prediction time: {time.time() - start:.3f} s")

    errors = np.abs(predictions - actual)
    mae = np.nanmean(errors)
    print(f"MAE: {mae:.6f}")

    if plot:
        idx = np.arange(n_test)
        plt.figure(figsize=(12, 8))
        plt.plot(idx, actual,       label='Actual',    linewidth=1)
        plt.plot(idx, predictions, '--', label='Predicted')
        plt.title('Actual vs Predicted')
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()

    return actual, predictions, errors
