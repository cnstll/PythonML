import numpy as np
from utils.normalization import zscore


def cross_validation(model, x: np.ndarray, y: np.ndarray, folds=10):
    try:
        init_thetas = {'thetas': model.thetas}
        x_folds = np.vsplit(x, folds)
        y_folds = np.vsplit(y, folds)
        print("Launching cross_validation...")
        scores = []
        x_train_shape = int(x.shape[0] - x.shape[0] / folds), x.shape[1]
        y_train_shape = int(y.shape[0] - y.shape[0] / folds), y.shape[1]
        for test_index in range(0, folds):
            model.set_params_(init_thetas)
            x_train = zscore(np.delete(x_folds, test_index,
                                       axis=0).reshape(x_train_shape))
            y_train = zscore(np.delete(y_folds, test_index,
                                       axis=0).reshape(y_train_shape))
            x_test = zscore(x_folds[test_index])
            y_test = zscore(y_folds[test_index])
            model.fit_(x_train, y_train)
            y_hat = model.predict_(x_test)
            scores.append(model.mse_(y_test, y_hat))
            print(f"{test_index}-th fold test fold completed")
        return np.mean(scores, dtype=float), model.thetas
    except Exception:
        return None, None
