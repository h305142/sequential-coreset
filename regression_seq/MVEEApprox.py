"""
Author: Murad Tukan
"""


import numpy as np
from scipy.optimize import approx_fprime
import cvxpy as cp
import scipy as sp
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
import numpy.linalg as la
import copy

class MVEEApprox(object):
    Epsilon = 1e-6

    def __init__(self, P, cost_func, maxIter=10, bound=1):
        self.cost_func = cost_func
        self.P = P
        self.bound = bound
        self.maxIter = maxIter
        self.c = np.zeros((P.shape[1], ))
        self.G = max(np.sqrt(P.shape[0]), np.max(np.sum(np.abs(P)**2,axis=-1)**(0.5))) * np.eye(P.shape[1], P.shape[1])
        self.oldG = copy.deepcopy(self.G)


    def separation_oracle(self, x):
        grad = approx_fprime(x, self.cost_func, self.Epsilon)
        return grad / np.linalg.norm(grad, np.inf)

    def obtain_concentric_ellipsoids(self):
        basis = np.eye(self.P.d)
        S = np.zeros(self.P.d, 2 * self.P.d)
        stop = False
        i = 0
        while not stop:
            v = basis[:,0]
            obj_func = (lambda x: cp.sum(cp.matmul(v.flatten(), x)))
            p_alpha = self.optimize_problem(obj_func)
            p_beta = self.optimize_problem(obj_func, operation=1)
            a = p_alpha - p_beta

            basis = sp.linalg.null_space(a.T)
            S[:,i] = p_alpha
            S[:,i+1] = p_beta
            i += 2

            if basis.size == 0 or i >= 2 * self.P.d:
                stop = True

    def optimize_problem(self, obj_func, operation=0):
        x = cp.Variable(self.P.d)
        constraints = [self.cost_func(x) <= self.bound]
        if operation == 0:
            objective = cp.Minimize(obj_func(x))
        else:
            objective = cp.Maximize(obj_func(x))

        problem = cp.Problem(objective, constraints)
        problem.solve()

        return x.value()

    def get_axis_points(self):
        U, s, vh = np.linalg.svd(self.G, full_matrices=True)
        # volume = np.prod(np.sqrt(s))
        d = s.shape[0]
        A = np.dot(np.diag(np.sqrt(s) / np.sqrt(d + 1)), U.T)
        points = np.vstack((A, -A))
        # points = np.tile(, vh.T)), d, axis=0)
        # temp = np.repeat(np.vstack((self.c[:, np.newaxis].T, -self.c[:, np.newaxis].T)), d, axis=0)
        temp = np.tile(self.c[:, np.newaxis].T, (2*d, 1))
        return points + temp

    def check_if_inside(self, P):
        vals = np.apply_along_axis(self.cost_func, 1, P)
        i = np.argmax(vals, axis=0)
        if vals[i] <= 1:
            return np.inf, vals[i]

        print('Maximal Value: {:.4f}'.format(vals[i]))

        return i, vals[i]

    def basic_ellipsoid_method(self):
        d = np.ma.size(self.P, axis=1)

        self.oldG = copy.deepcopy(self.G)
        while self.cost_func(self.c) > 1 :
            H = self.separation_oracle(self.c)
            b = np.dot(self.G, H) / np.sqrt(np.abs(np.dot(H, np.dot(self.G, H))))
            self.c = self.c - 1.0 / (d + 1.0) * b
            self.G = d ** 2.0 / (d ** 2.0 - 1.0) * (self.G - (2.0 / (d + 1.0)) * np.dot(b[:, np.newaxis], b[:, np.newaxis].T))

        if not self.isPD(self.G):
            print('Corrected back to PSD at Basic ellipsoid method')
            self.G = self.nearestPD(self.G)


    def shallow_cut_update(self, point):
        d = np.ma.size(self.G, 0)
        rho = 1.0 / (d + 1.0) ** 2.0
        sigma = d ** 3.0 * (d + 2.0) / ((d + 1) ** 3.0 * (d - 1.0))
        zeta = 1.0 + 1.0 / (2.0 * d ** 2.0 * (d + 1.0) ** 2.0)
        tau = 2.0 / ((d + 1.0) * d)

        b = np.dot(self.G, point) / np.sqrt(np.abs(np.dot(point, np.dot(self.G, point))))
        self.oldG = copy.deepcopy(self.G)
        self.G = zeta * sigma * (self.G - tau * np.dot(b[:, np.newaxis], b[:, np.newaxis].T))
        self.c = self.c - rho * b

        if not self.isPD(self.G):
            print('Corrected back to PSD at Shallow cut update')
            self.G = self.nearestPD(self.G)

    def compute_approximated_MVEE(self):
        stop = False
        iter = 0
        while not stop:
            self.basic_ellipsoid_method()
            axis_points = self.get_axis_points()
            i, val = self.check_if_inside(axis_points)
            if np.isinf(i):
                stop = True
            else:
                sep_grad = self.separation_oracle(axis_points[i, :])
                self.shallow_cut_update(sep_grad)

            if iter > self.maxIter:
                self.G = self.G / val
                iter = 0
                print('HMM')
                continue
            iter += 1

        E = np.linalg.cholesky(self.G)
        return E, self.c

    @staticmethod
    def nearestPD(A):
        """Find the nearest positive-definite matrix to input

        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
        credits [2].

        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
        """

        B = (A + A.T) / 2
        _, s, V = la.svd(B)

        H = np.dot(V.T, np.dot(np.diag(s), V))

        A2 = (B + H) / 2

        A3 = (A2 + A2.T) / 2

        if MVEEApprox.isPD(A3):
            return A3

        spacing = np.spacing(la.norm(A))
        # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
        # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
        # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
        # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
        # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
        # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
        # `spacing` will, for Gaussian random matrixes of small dimension, be on
        # othe order of 1e-16. In practice, both ways converge, as the unit test
        # below suggests.
        I = np.eye(A.shape[0])
        k = 1
        while not MVEEApprox.isPD(A3):
            mineig = np.min(np.real(la.eigvals(A3)))
            A3 += I * (-mineig * k ** 2 + spacing)
            k += 1

        return A3

    @staticmethod
    def isPD(B):
        """Returns true when input is positive-definite, via Cholesky"""
        try:
            _ = la.cholesky(B)
            return True
        except la.LinAlgError:
            return False

    @staticmethod
    def plotEllipsoid(center, radii, rotation, ax=None, plotAxes=True, cageColor='r', cageAlpha=1):
        """Plot an ellipsoid"""
        make_ax = (ax is None)
        if make_ax:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        u = np.linspace(0.0, 2.0 * np.pi, 100)
        v = np.linspace(0.0, np.pi, 100)

        # cartesian coordinates that correspond to the spherical angles:
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
        # rotate accordingly
        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = np.dot(np.array([x[i, j], y[i, j], z[i, j]]), rotation) + center.flatten()

        if plotAxes:
            # make some purdy axes
            axes = np.array([[radii[0], 0.0, 0.0],
                             [0.0, radii[1], 0.0],
                             [0.0, 0.0, radii[2]]])
            # rotate accordingly
            for i in range(len(axes)):
                axes[i] = np.dot(axes[i], rotation)

            print('Axis are: ', axes)
            # print(axes + center.flatten())

            # plot axes
            print('Whole points are: ')
            for p in axes:
                X3 = np.linspace(-p[0], p[0], 2) + center[0]
                Y3 = np.linspace(-p[1], p[1], 2) + center[1]
                Z3 = np.linspace(-p[2], p[2], 2) + center[2]
                ax.plot3D(X3, Y3, Z3, color='m')
                PP = np.vstack((X3, Y3, Z3)).T
                print(PP)

        # plot ellipsoid
        ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color=cageColor, alpha=cageAlpha)
        plt.show()

    def plotBodyAndEllips(self, B, E):
        N = 10000
        U, D, V = np.linalg.svd(E, full_matrices=True)
        a = D[0]
        b = D[1]
        theta = np.expand_dims(np.arange(start=0, step=1.0 / N, stop=2.0 * np.pi + 1.0 / N), 1).T

        state = np.vstack((a * np.cos(theta), b * np.sin(theta)))
        X = np.dot(U, state) + self.c[:, np.newaxis]

        ax = plt.subplot(111)
        plt.plot(X[0, :], X[1, :], color='black', linewidth=5)

        vals = np.apply_along_axis(lambda x: np.linalg.norm(x.flatten() - self.c.flatten()), 0, X)
        i = np.argmax(vals)

        print(X[:, i])

        plt.scatter(self.c[0], self.c[1], marker='+', color='green')
        plt.grid(True)

        # hull = ConvexHull(B)
        # for simplex in hull.simplices:
        #     plt.plot(B[simplex, 0], B[simplex, 1], 'k-')

        # plt.scatter(B[:, 0], B[:, 1], marker='D', color='orange')

        # plt.scatter(self.c[0], self.c[1], marker='^', color='green')
        # plt.scatter(X[0, i], X[1, i], marker='*', color='black')
        plt.scatter(B[:, 0], B[:, 1], marker='*', color='green')
        plt.show()

    @staticmethod
    def main():
        P = np.random.rand(10000, 400)
        cost_func = lambda x: np.linalg.norm(np.dot(P, x), ord=1)
        tol = 1/100
        start_time = time.time()

        mvee = MVEEApprox(P, cost_func, maxIter=10)
        E, C = mvee.compute_approximated_MVEE()

        print('Ellip took {:.4f}'.format(time.time() - start_time))
        if P.shape[1] <= 3:
            N = 1000
            X = np.random.randn(N, P.shape[1])
            vals = np.apply_along_axis(cost_func, 1, X)
            X = np.multiply(X, 1.0 / vals[:, np.newaxis])
            if P.shape[1] == 2:
                mvee.plotBodyAndEllips(X, E)
            else:
                fig = plt.figure()
                ax = plt.axes(projection='3d')

                # from scipy.spatial import ConvexHull
                # hull = ConvexHull(X)

                # Plot defining corner points
                # ax.plot(X.T[0], X.T[1], X.T[2], "ko")
                # for s in hull.simplices:
                #     s = np.append(s, s[0])  # Here we cycle back to the first coordinate
                #     ax.plot(X[s, 0], X[s, 1], X[s, 2], "b-")
                ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], color='black', marker='o')
                U, D, V = la.svd(E, full_matrices=True)
                mvee.plotEllipsoid(C, D, U.T, ax=ax)


if __name__ == '__main__':
    MVEEApprox.main()