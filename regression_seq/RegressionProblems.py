import numpy as np
import scipy as sp
import cvxpy as cp
import time
from MainProgram import Utils
import PointSet


class RegressionProblem(object):
    def __init__(self, p_norm=2):
        assert(p_norm < 0, 'p_norm must be a positive scalar!')
        self.p_norm = p_norm
        self.coef_ = None
        # self.time_taken = None
        self.intercept_ = None

    def fit(self, X, Y, weights):
        d = X.shape[1]
        start_time = time.time()
        if self.p_norm < 1 or 'res' in Utils.PROBLEM_TYPE:
            func = lambda x: np.sum(np.multiply(weights, np.abs(np.dot(X, x) - Y) ** self.p_norm)) if self.p_norm < 1 \
                else np.sum(np.multiply(weights, np.min(np.abs(np.dot(X, x) - Y), np.linalg.norm(x, Utils.Z))))
            grad = lambda x: sp.optimize.approx_fprime(x, func, Utils.EPSILON)
            optimal_x = None
            optimal_val = np.Inf
            for i in range(Utils.OPTIMIZATION_NUM_INIT):
                x0 = Utils.createRandomInitialVector(d)
                res = sp.optimize.minimize(fun=func, x0=x0, jac=grad, method='L-BFGS-B')
                temp = Utils.OBJECTIVE_COST(PointSet.PointSet(np.hstack((X,Y)), weights), res.x)
                if temp < optimal_val:
                    optimal_val = temp
                    optimal_x = res.x
            self.coef_ = optimal_x
        else:
            w = cp.Variable(d, )

            loss = cp.sum(cp.multiply(weights, cp.power(cp.abs(cp.matmul(X, w)- Y), self.p_norm)))
            constraints = []

            prob = cp.Problem(cp.Minimize(loss), constraints)
            prob.solve()
            self.coef_ = w.value()
        # self.time_taken = time.time() - start_time
