import numpy as np
from sklearn.metrics import auc, roc_curve

y = np.array([np.random.randint(0,2) for _ in range(10)])

y_hat = np.array([np.random.random() for _ in range(10)])


fpr, tpr, threshold = roc_curve(y, y_hat, pos_label=1)
sklearn_auc = auc(fpr, tpr)

print("sklearn auc", sklearn_auc)

def my_auc_impl():

    p = sum(y)
    n = len(y) - p
    threshold = sorted(list(set(y_hat)), reverse=True)
    area = 0
    last_tpr = 0
    last_fpr = 0
    for thres in threshold:
        y_hat_pos_idx = np.where(y_hat >= thres)
        tp = sum(y[y_hat_pos_idx])
        fp = len(y[y_hat_pos_idx]) - tp
        tpr = tp / p
        fpr = fp / n
        area += ((tpr + last_tpr) * (fpr-last_fpr) / 2)
        last_tpr = tpr
        last_fpr = fpr

    return area

print("my impl auc", my_auc_impl())
