import numpy as np
import bottleneck as bn
from concurrent.futures import ThreadPoolExecutor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor

def target_calculation(alphas, last_idx, X):
    values = {}
    for alpha in alphas:
        idxs = (last_idx - np.cumsum((alpha[1:])[::-1])[::-1])
        values[tuple(alpha)] = X[idxs]
    return values

trees = {}

def classifier_prediction(y_alphas):
    preds = []
    for alpha, vec in y_alphas.items():
        try:
            p = trees[alpha].predict(vec.reshape(1, -1))[0]
        except Exception:
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

    with ThreadPoolExecutor() as executor:
        for alpha, model in executor.map(fit_tree, z.keys()):
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

def train_classifier(alphas, z, X, classifier_points, threshold):
    preds_matrix = []
    start = len(X) - classifier_points
    for alpha in alphas:
        assert alpha in trees, f"Tree for {alpha} not trained"

    for i in range(start, len(X)):
        y_alphas = target_calculation(alphas, i, X)
        vec = classifier_prediction(y_alphas).reshape(-1)
        vec = np.where(np.isnan(vec), np.nanmean(vec), vec)
        preds_matrix.append(vec)

    X_preds = np.array(preds_matrix)
    y_labels = np.array([
        int(abs(np.nanmean(row) - X[start + idx]) > threshold)
        for idx, row in enumerate(X_preds)
    ])

    clf = DecisionTreeClassifier(min_samples_leaf=1, random_state=42)
    clf.fit(X_preds, y_labels)
    return clf