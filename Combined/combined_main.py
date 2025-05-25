from data_generator import generate_lorenz_x
from combined_runner2 import run_combined_prediction


x = generate_lorenz_x()
print("Длина ряда:", len(x))


n_train = 6267
n_test = 50


actual, preds, labels = run_combined_prediction(
    series=x,
    n_train=n_train,
    n_test=n_test,
    l=4,
    k=10,
    combiner_points=2500,
    threshold=0.1,
    plot=True
)

#500 норм