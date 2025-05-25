import numpy as np
from data_generator import generate_lorenz_x
from cycle_runner import run_combined_prediction

x = generate_lorenz_x()
print("Длина ряда:", len(x))

n_train = 6267
n_test  = 50

actual, preds, errors = run_combined_prediction(
    series=x,
    n_train=n_train,
    n_test=n_test,
    l=4,
    k=10,
    threshold=0.01,
    plot=True
)

mse = np.nanmean((preds - actual)**2)
print(f"MSE: {mse:.6f}")
