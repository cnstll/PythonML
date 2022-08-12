import numpy as np
import sklearn.metrics as m


def accuracy_score_(y, y_hat):
    """
    Compute the accuracy score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
    Returns:
        The accuracy score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    try:
        if any(not isinstance(p, np.ndarray) for p in (y, y_hat)):
            return None
        if any(p.size == 0 for p in (y, y_hat)):
            return None
        if y.shape != y_hat.shape:
            return None
        total_predictions = y_hat.size
        right_predictions = np.sum(y == y_hat)
        return float(right_predictions / total_predictions)
    except Exception:
        return None


def precision_score_(y, y_hat, pos_label=1):
    """
    Compute the precision score. Control false positive.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to
        report the precision_score (default=1)
    Returns:
        The precision score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    try:
        if any(not isinstance(p, np.ndarray) for p in (y, y_hat)):
            return None
        if any(p.size == 0 for p in (y, y_hat)):
            return None
        if y.shape != y_hat.shape:
            return None
        if any(pos_label not in p for p in (y, y_hat)):
            return None
        true_positive = np.sum((y == y_hat) & (y == pos_label))
        false_positive = np.sum((y != y_hat) & (y_hat == pos_label))
        return float(true_positive / (false_positive + true_positive))
    except Exception:
        return None


def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score. Control for false negatives.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report
        the precision_score (default=1)
    Returns:
        The recall score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    try:
        if any(not isinstance(p, np.ndarray) for p in (y, y_hat)):
            return None
        if any(p.size == 0 for p in (y, y_hat)):
            return None
        if y.shape != y_hat.shape:
            return None
        if any(pos_label not in p for p in (y, y_hat)):
            return None
        true_positive = np.sum((y == y_hat) & (y == pos_label))
        false_negative = np.sum((y != y_hat) & (y_hat != pos_label))
        return float(true_positive / (false_negative + true_positive))
    except Exception:
        return None


def f1_score_(y, y_hat, pos_label=1):
    """
    Compute the f1 score.
    Control both False positives and False negatives
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report
        the precision_score (default=1)
    Returns:
        The f1 score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    p_score = precision_score_(y, y_hat, pos_label)
    r_score = recall_score_(y, y_hat, pos_label)
    if any(p is None for p in (p_score, r_score)):
        return None
    num = 2 * p_score * r_score
    denum = p_score + r_score
    return float(num / denum)


if __name__ == '__main__':
    print("> Example 1")
    y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))
    # Accuracy
    # your implementation
    print(accuracy_score_(y, y_hat))
    # sklearn implementation
    print(m.accuracy_score(y, y_hat))

    # Precision score
    # your implementation
    print(precision_score_(y, y_hat))
    # sklearn implementation
    print(m.precision_score(y, y_hat))

    # Recall score
    # your implementation
    print(recall_score_(y, y_hat))
    # sklearn implementation
    print(m.recall_score(y, y_hat))

    # F1score
    # your implementation
    print(f1_score_(y, y_hat))
    # sklearn implementation
    print(m.f1_score(y, y_hat))
    print()

    print("> Example 2")
    y_hat = np.array(['norminet', 'dog', 'norminet',
                      'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet',
                  'dog', 'norminet', 'dog', 'norminet'])
    # Accuracy
    # your implementation
    print(accuracy_score_(y, y_hat))
    # sklearn implementation
    print(m.accuracy_score(y, y_hat))

    # Precision score
    # your implementation
    print(precision_score_(y, y_hat, pos_label='dog'))
    # sklearn implementation
    print(m.precision_score(y, y_hat, pos_label='dog'))

    # Recall score
    # your implementation
    print(recall_score_(y, y_hat, pos_label='dog'))
    # sklearn implementation
    print(m.recall_score(y, y_hat, pos_label='dog'))

    # F1score
    # your implementation
    print(f1_score_(y, y_hat, pos_label='dog'))
    # sklearn implementation
    print(m.f1_score(y, y_hat, pos_label='dog'))
    print()

    print("> Example 3")
    y_hat = np.array(['norminet', 'dog', 'norminet',
                      'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet',
                  'dog', 'norminet', 'dog', 'norminet'])
    # Accuracy
    # your implementation
    print(accuracy_score_(y, y_hat))
    # sklearn implementation
    print(m.accuracy_score(y, y_hat))

    # Precision score
    # your implementation
    print(precision_score_(y, y_hat, pos_label='norminet'))
    # sklearn implementation
    print(m.precision_score(y, y_hat, pos_label='norminet'))

    # Recall score
    # your implementation
    print(recall_score_(y, y_hat, pos_label='norminet'))
    # sklearn implementation
    print(m.recall_score(y, y_hat, pos_label='norminet'))

    # F1score
    # your implementation
    print(f1_score_(y, y_hat, pos_label='norminet'))
    # sklearn implementation
    print(m.f1_score(y, y_hat, pos_label='norminet'))
    print()
