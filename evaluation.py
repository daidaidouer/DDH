import numpy as np
from sklearn import metrics

def get_map(recall, re_codes, re_labels, q_codes, q_labels):
    all_rel = np.dot(q_codes, re_codes.T)
    ids = np.argsort(-all_rel, 1)
    map = np.zeros(q_codes.shape[0], dtype=np.float)
    print("#calc maps# calculating maps")
    for i in range(q_labels.shape[0]):
        idx = ids[i]
        score = all_rel[i, idx[:recall]]
        gt = np.zeros(recall, dtype=np.float)
        for j in range(len(gt)):
            gt[j] = np.max(re_labels[idx[j]] + q_labels[i]) - 1
        map[i] = metrics.average_precision_score(y_true=gt, y_score=score)
    maps = np.mean(map)
    print("maps: ", maps)
    return maps
