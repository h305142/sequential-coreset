import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import RegressionProblems as RP
import time
from multiprocessing import Lock


class Optimizor(object):
    MODELS = {
        'logistic': lambda C, tol, Z: LogisticRegression(tol=tol, C=C, solver='lbfgs', max_iter=1e4),
        'svm': lambda C, tol, Z: SVC(kernel='linear', C=C, tol=tol),
        'lz': lambda C, tol, Z: RP.RegressionProblem(Z)
    }

    # create mutex for multi-threading purposes
    mutex = Lock()

    def __init__(self, P, problem_type, C, tol, Z, objective_cost):
        self.problem_type = problem_type
        self.model = Optimizor.MODELS[problem_type](C=C, tol=tol, Z=Z)
        self.sum_weights = None
        self.C = C
        self.Z = Z
        self.TOL = tol
        self.objective_cost = objective_cost
        self.P = P
        self.optimal_w = None

    def defineSumOfWegiths(self, W):
        self.sum_weights = np.sum(W)

    def fit(self, P):
        start_time = time.time()
        if 'lz' not in self.problem_type:
            Optimizor.mutex.acquire()
            c_prime = self.model.C * float(self.sum_weights / (np.sum(P.W)))
            params = {"C": c_prime}
            self.model.set_params(**params)
            Optimizor.mutex.release()

        self.model.fit(P.P[:, :-1], P.P[:, -1], P.W)
        Optimizor.mutex.acquire()
        w, b = self.model.coef_, self.model.intercept_
        sol = np.hstack((w.flatten(), b)) if b is not None else w
        if self.optimal_w is None:
            self.optimal_w = sol
        Optimizor.mutex.release()
        return self.computeCost(self.P, sol), time.time() - start_time

    def computeCost(self, P, x):
        return self.objective_cost(P, x, (self.sum_weights, ))