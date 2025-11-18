import numpy as np

def ft_log_loss(y_true, y_pred, eps=1e-15):
    """
    y_true : (n_samples, n_classes)
    y_pred : (n_samples, n_classes)
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]


def ft_accuracy_score(y_true_labels, y_pred_labels):
    return np.mean(y_true_labels == y_pred_labels)