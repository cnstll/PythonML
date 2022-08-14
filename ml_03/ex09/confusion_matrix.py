import numpy as np
from sklearn.metrics import confusion_matrix


def confusion_matrix_(y_true, y_hat, labels=None):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
    y_true: numpy.ndarray for the correct labels
    y_hat: numpy.ndarray for the predicted labels
    labels: Optional, a list of labels to index the matrix.
    This may be used to reorder or select a subset of labels. (default=None)
    Returns:
    The confusion matrix as a numpy ndarray.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if any(not isinstance(p, np.ndarray) for p in (y_true, y_hat)):
        return None
    if any(p.size == 0 for p in (y_true, y_hat)):
        return None
    if y_true.shape != y_hat.shape:
        return None
    if labels is None:
        labels = np.unique(y_hat)
    confusion_mat = np.empty((len(labels), len(labels)), dtype=np.int32)
    for i, l in enumerate(labels):
        pos = np.sum((y_true == y_hat) & (y_true == l))
        other_labels = [k for k in labels if k != l]
        neg = np.zeros(len(other_labels))
        for j, al in enumerate(other_labels):
            neg[j] = np.sum((y_true != y_hat)
                            & (y_true == l) & (y_hat == al))
        confusion_mat[i] = np.insert(neg, i, pos)
    return confusion_mat


if __name__ == '__main__':
    y_hat = np.array([['norminet'], ['dog'], ['norminet'],
                     ['norminet'], ['dog'], ['bird']])
    y = np.array([['dog'], ['dog'], ['norminet'], [
        'norminet'], ['dog'], ['norminet']])

    # Your implementation
    print(confusion_matrix_(y, y_hat))
    # Sklearn implementation
    print(confusion_matrix(y, y_hat))

    # Your implementation
    print(confusion_matrix_(y, y_hat, labels=['dog', 'norminet']))
    # Sklearn implementation
    print(confusion_matrix(y, y_hat, labels=['dog', 'norminet']))
