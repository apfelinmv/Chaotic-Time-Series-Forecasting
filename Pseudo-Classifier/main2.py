import matplotlib.pyplot as plt
from data_generator import generate_lorenz_x
from combined_runner import run_combined_prediction

x = generate_lorenz_x()
print("Длина ряда:", len(x))

n_train = 6267
n_test  = 50

actual, preds, labels, mse = run_combined_prediction(
    series=x,
    n_train=n_train,
    n_test=n_test,
    l=4,
    k=10,
    combiner_points=500,
    threshold=0.1,
    plot=True
)
