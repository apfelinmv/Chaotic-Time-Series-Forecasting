import matplotlib.pyplot as plt
from data_generator import generate_lorenz_x
from combined_runner import run_combined_prediction

# 1) Генерируем ряд
x = generate_lorenz_x()
print("Длина ряда:", len(x))

# 2) Параметры обучения и теста
n_train = 6267
n_test  = 50

# 3) Запуск комбинированного прогноза (plot=False, чтобы рисовать самим)
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

#500 норм