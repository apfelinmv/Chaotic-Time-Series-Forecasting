import numpy as np
import bottleneck as bn
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
def target_calculation(alphas, last_idx, X):
    values = {}
    for alpha in alphas:
        target_idxs = (last_idx - np.cumsum((alpha[1:])[::-1])[::-1])
        values[tuple(alpha)] = X[target_idxs]
    return values

trees = {}

def classifier_prediction(y_alphas):
    preds = []
    for alpha, vec in y_alphas.items():
        try:
            p = trees[alpha].predict(vec.reshape(1, -1))[0]
        except Exception:
            print(alpha)
            p = np.nan
        preds.append(p)
    return np.array(preds).reshape(1, -1)


def partial_model(z, classifier_points):
    def fit_tree(alpha):
        data = z[alpha]
        train = data[:-classifier_points]
        X_train, y_train = train[:, :-1], train[:, -1]
        model = DecisionTreeRegressor(min_samples_leaf=1,
                                      random_state=42,
                                      max_depth=365,
                                      splitter='best')
        model.fit(X_train, y_train)
        return alpha, model

    futures = []
    with ThreadPoolExecutor() as executor:
        for alpha in z.keys():
            futures.append(executor.submit(fit_tree, alpha))
        for future in tqdm(as_completed(futures), total=len(futures), desc="Training trees"):
            alpha, model = future.result()
            trees[alpha] = model

def train_combiner(alphas, X, combiner_points):
    features, targets = [], []

    start = len(X) - combiner_points
    for i in range(start, len(X)):
        y_alphas = target_calculation(alphas, i, X)
        vec = classifier_prediction(y_alphas).reshape(-1)
        features.append(vec)
        targets.append(X[i])

    X_feat = np.array(features)
    y_true = np.array(targets)


    combiner = RandomForestRegressor(n_estimators=1, random_state=42)
    combiner.fit(X_feat, y_true)
    return combiner

