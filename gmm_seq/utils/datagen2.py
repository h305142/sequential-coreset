import numpy as np
from sklearn.datasets import make_blobs, make_classification

def gen_crp(n, d, n_components, *, method = 2, cluster_std=120, random_state=None):
    if method == 1:
        # make_classification, we cannot add noise when d is small in order to
        # avoid the ValueError: n_classes * n_clusters_per_class must be smaller or equal 2**n_informative
        if d < 8:

            X, target_label = make_classification(n_samples=n, n_features=d, n_informative=d,
                                                  n_redundant=0, n_repeated=0,
                                                  n_classes=n_components,
                                                  n_clusters_per_class=1, weights=None, flip_y=0.01, class_sep=1.0,
                                                  hypercube=False, shift=0.0,
                                                  scale=1.0, shuffle=True, random_state=random_state)
        else:
            d_informative = int(np.floor(d / 2))
            d_redundant = int(np.floor(d / 4))
            d_r = d - d_informative - d_redundant
            X, target_label = make_classification(n_samples=n, n_features=d, n_informative=d_informative,
                                                  n_redundant=d_redundant, n_repeated=d_redundant,
                                                  n_classes=n_components,
                                                  n_clusters_per_class=1, weights=None, flip_y=0.01, class_sep=1.0,
                                                  hypercube=False, shift=0.0,
                                                  scale=1.0, shuffle=True, random_state=random_state)


    elif method == 2:
        centers = np.random.rand(n_components, d) * 1000
        X, target_label = make_blobs(n_samples=n, centers=centers, n_features=d,
                                     random_state=random_state, cluster_std=120)
        # add perturbation
        transformation = np.random.random((d, d))
        X = np.dot(X, transformation)

    return X, target_label