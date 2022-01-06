import numpy as np
import MVEEApprox


class PointSet(object):
    def __init__(self, P, W=None, ellipsoid_max_iters=10, problem_type=None, use_svd=False, compute_U=True):
        self.P = P
        if P is not None:
            self.n, self.d = P.shape
            self.W = np.ones((self.n, )) if W is None else W
        else:
            self.n = self.d = 0
            self.W = W
        self.U = self.D = self.V = None
        self.mvee = None
        self.ellipsoid_max_iters = ellipsoid_max_iters
        self.cost_func = lambda x: np.linalg.norm(np.multiply(self.W, np.dot(self.P, x), ord=1))
        self.problem_type = problem_type
        self.use_svd = use_svd

        if problem_type is not None and 'lz' not in problem_type and self.P is not None:
            self.pos_idxs = np.where(self.P[:, -1] > 0)[0]
            self.neg_idxs = np.setdiff1d(np.arange(self.n), self.pos_idxs)
            self.sum_weights_pos = np.sum(self.W[self.pos_idxs])
            self.sum_weights_neg = np.sum(self.W[self.neg_idxs])
            self.sum_weights = self.sum_weights_neg + self.sum_weights_pos

        if self.P is not None and compute_U:
            self.computeU()

    def computeU(self):
        self.U = np.empty(self.P[:, :-1].shape)
        if self.use_svd:
            if 'lz' not in self.problem_type:
                self.U[self.pos_idxs, :], _, _ = \
                    np.linalg.svd(np.multiply(np.sqrt(self.W[self.pos_idxs])[:, np.newaxis],
                                              self.P[self.pos_idxs, :-1]), full_matrices=False)
                self.U[self.neg_idxs, :], _, _ = \
                    np.linalg.svd(np.multiply(np.sqrt(self.W[self.neg_idxs])[:, np.newaxis],
                                              self.P[self.neg_idxs, :-1]), full_matrices=False)
            else:
                self.U, _, _ = np.linalg.svd(np.multiply(np.sqrt(self.W)[:, np.newaxis], self.P),
                                                       full_matrices=False)
        else:
            self.mvee = MVEEApprox.MVEEApprox(self.P, self.cost_func, self.ellipsoid_max_iters)
            ellipsoid, _ = self.mvee.compute_approximated_MVEE()
            _, self.D, self.V = np.linalg.svd(np.linalg.pinv(ellipsoid), full_matrices=True)
            self.U = np.dot(self.P, np.linalg.inv(self.D.dot(self.V)))

    def mergePointSet(self, Q):
        self.P = np.vstack((self.P, Q.P))
        self.W = np.hstack((self.W, Q.W))
        self.n, self.d = self.P.shape
        self.computeU()

    # def computeSensitivity(self):
    #     pass

