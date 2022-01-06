"""*****************************************************************************************
MIT License
Copyright (c) 2020 Murad Tukan, Alaa Maalouf, Dan Feldman
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*****************************************************************************************"""
import os
import scipy as sp
import numpy as np
import time
import pathlib
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime
from scipy.stats import ortho_group
import cvxpy as cp
import copy
from sklearn.preprocessing import Normalizer
import PointSet


################################################## General constants ###################################################
EPSILON = 1e-9  # used for approximated the gradient of a loss function
TOL = 0.01  # defines the approximation with respect to the minimum volume enclosing ellipsoid
ELLIPSOID_MAX_ITER = 10
OPTIMIZATION_TOL = 1e-6
OPTIMIZATION_NUM_INIT = 10
Z = 2  # the norm
LAMBDA = 1  # the regularization parameter
PARALLELIZE = False  # whether to apply the experimental results in a parallel fashion
THREAD_NUM = 4
DATA_FOLDER = 'datasets'

# M estimator loss functions supported by our framework
SENSE_BOUNDS = {
    'logistic': (lambda x, w, args=None: 32 / LAMBDA * (2 / args[0] + w * np.linalg.norm(x, ord=2, axis=1) ** 2)
                                          * args[0]),
    'nclz': (lambda x, w, args=None: w * np.linalg.norm(x, ord=Z, axis=1) ** Z),
    'svm': (lambda x, w, args=None: np.maximum(9 * w / args[0], 2 * w / args[1]) + 13 * w / (4 * args[0]) +
                              125 * (args[0] + args[1]) / (4 * LAMBDA) * (w * np.linalg.norm(x, ord=2, axis=1)**2 +
                                                                          w/(args[0] + args[1]))),
    'restricted_lz': (lambda x, w, args=None: w * np.minimum(np.linalg.norm(x, ord=2, axis=1),
                                                    args[1] ** np.abs(0.5 - 1/Z) * np.linalg.norm(args[0]))),
    'lz': (lambda x, w, args=None: w * np.linalg.norm(x, ord=Z, axis=1) ** Z if 1 <= Z <= 2
            else args[0]**(Z/2) * w * np.linalg.norm(x, ord=Z, axis=1)**Z),
    'lse': (lambda x, w, args=None: w * np.linalg.norm(x, ord=1, axis=1))
}

OBJECTIVE_COSTS = {
    'logistic':
        (lambda P, x, args=None: (np.sum(P.W) / (2 * args[0]) if args is not None else (1 / 2)) *
                                 np.linalg.norm(x[:-1], 2) ** 2 +
                                 LAMBDA * np.sum(np.multiply(P.W, np.log1p(np.exp(-np.multiply(P.P[:, -1],
                                                                                        np.dot(P.P[:, :-1], x[:-1])
                                                                                        + x[-1])))))),
    'svm':
        (lambda P, x, args=None: (np.sum(P.W) / (2 * args[0]) if args is not None else (1 / 2)) *
                                 np.linalg.norm(x[:-1], 2) ** 2 +
                                 LAMBDA * np.sum(np.multiply(P.W, np.maximum(0,
                                                                         1 - (np.multiply(P.P[:, -1],
                                                                                          np.dot(P.P[:, :-1], x[:-1])
                                                                                          + x[-1])))))),
    'lz':
        (lambda P, x, args=None: np.sum(np.multiply(P.W, np.abs(np.dot(P.P[:, :-1], x) - P.P[:, -1]) ** Z))),
'restricted_lz':
        (lambda P, x, args=None: np.sum(np.multiply(P.W, np.minimum(np.sum(np.abs(P.P, x), 1), np.linalg.norm(x, Z))))),
    'lse':
        (lambda P, x, args=None: np.linalg.norm(P.P - x, 'fro') ** 2)
}


############################################# Data option constants ####################################################
SYNTHETIC_DATA = 0  # use synthetic data
REAL_DATA = 1  # use real data
DATA_TYPE = REAL_DATA  # distinguishes between the use of real vs synthetic data

########################################### Experimental results constants #############################################
# colors for our graphs
COLOR_MATCHING = {'Our coreset': 'red',
                  'Uniform sampling': 'blue',
                  'All data': 'black'}

REPS = 1  # number of repetitions for sampling a coreset
SEED = np.random.randint(1, int(1e7), REPS)  # Seed for each repetition
NUM_SAMPLES = 10  # number of coreset sizes
x0 = None  # initial solution for hueristical solver
OBJECTIVE_COST = None
PROBLEM_TYPE = None
SENSE_BOUND = None
USE_SVD = False
PREPROCESS_DATA = False


def initializaVariables(problem_type, z=2, Lambda=1):
    global Z, LAMBDA, OBJECTIVE_COST, PROBLEM_TYPE, SENSE_BOUND, USE_SVD, PREPROCESS_DATA
    Z = z
    LAMBDA = Lambda
    OBJECTIVE_COST = OBJECTIVE_COSTS[problem_type]  # the objective function which we want to generate a coreset for
    PROBLEM_TYPE = problem_type
    SENSE_BOUND = SENSE_BOUNDS[problem_type]
    if ('lz' in problem_type and z != 2) or problem_type == 'lse':
        USE_SVD = False
        PREPROCESS_DATA = False
    else:
        USE_SVD = True
        PREPROCESS_DATA = True

    var_dict = {}

    variables = copy.copy(list(globals().keys()))

    for var_name in variables:
        if var_name.isupper():
            var_dict[var_name] = eval(var_name)

    return var_dict



def preprocessData(P):
    global PREPROCESS_DATA

    if PREPROCESS_DATA:
        y = P[:, -1]
        min_value = np.min(y)
        max_value = np.max(y)
        P = Normalizer().fit_transform(P[:, :-1], P[:, -1])

        y[np.where(y == min_value)[0]] = -1
        y[np.where(y == max_value)[0]] = 1
        P = np.hstack((P, y[:, np.newaxis]))

    return P






# def getObjectiveFunction():
#     global PROBLEM_DEF, OBJECTIVE_FUNC, GRAD_FUNC
#
#     if PROBLEM_DEF == 1:
#         OBJECTIVE_FUNC = (lambda P, w: np.sum(np.multiply(P.W,np.log(1.0 + np.square(np.dot(P.P[:, :-1], w) + P.P[:, -1])))))
#         GRAD_FUNC = (lambda P, w: np.sum(
#             np.multiply(P.W, np.multiply(np.expand_dims(2/(1.0 + np.square(np.dot(P.P[:, :-1], w) - P.P[:, -1])), 1),
#                                        np.multiply(P.P[:, :-1], np.expand_dims(np.dot(P.P[:, :-1], w) + P.P[:, -1], 1)), 0))))


def generateSampleSizes(n):
    """
    The function at hand, create a list of samples which denote the desired coreset sizes.

    :param n: An integer which denotes the number of points in the dataset.
    :return: A list of coreset sample sizes.
    """
    global NUM_SAMPLES

    min_val = int(2 * np.log(n) ** 2)  # minimum sample size
    max_val = int(6 * n ** 0.6)  # maximal sample size
    samples = np.geomspace([min_val], [max_val], NUM_SAMPLES)  # a list of samples
    return samples


# def readSyntheticRegressionData():
#     data = np.load('SyntheticRegDataDan.npz')
#     X = data['X']
#     y = data['y']
#     P = PointSet(np.hstack((X[:, np.newaxis], -y[:, np.newaxis])))
#     return P


def plotPointsBasedOnSens():
    sens = np.load('sens.npy')
    data = np.load('SyntheticRegDataDan.npz')
    X = data['X']
    y = data['y']
    # getObjectiveFunction()
    P = np.hstack((X[:, np.newaxis], -y[:, np.newaxis]))

    colorbars = ['bwr']#, 'seismic', 'coolwarm', 'jet', 'rainbow', 'gist_rainbow', 'hot', 'autumn']

    for i in range(len(colorbars)):
        plt.style.use('classic')
        min_, max_ = np.min(sens), np.max(sens)

        plt.scatter(P[:, 0], P[:, 1], c=sens, marker='o', s=50, cmap=colorbars[i])
        plt.clim(min_, max_)

        ax = plt.gca()
        cbar = plt.colorbar(pad=-0.1, fraction=0.046)
        cbar.ax.get_yaxis().labelpad = 24
        cbar.set_label('Sensitivity', rotation=270, size='xx-large', weight='bold')
        cbar.ax.tick_params(labelsize=24)
        plt.ylabel('y')
        plt.xlabel('x')
        plt.axis('off')
        figure = plt.gcf()
        figure.set_size_inches(20, 13)
        plt.savefig('Sens{}.pdf'.format(i), bbox_inches='tight', pad_inches=0)


def createRandomRegressionData(n=2e4, d=2):
    X, y = sklearn.datasets.make_regression(n_samples=int(n), n_features=d, random_state=0, noise=4.0,
                           bias=100.0)

    # X = np.random.randn(int(n),d)
    # y = np.random.rand(y.shape[0], )

    X = np.vstack((X, 1000 * np.random.rand(20, d)))
    y = np.hstack((y, 10000 * np.random.rand(20, )))

    np.savez('SyntheticRegData', X=X, y=y)


def plotEllipsoid(center, radii, rotation, ax=None, plotAxes=True, cageColor='r', cageAlpha=1):
    """Plot an ellipsoid"""
    make_ax = ax == None
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

##################################################### READ DATASETS ####################################################

def readRealData(dataset, problemType=0):
    """
    This function, given a physical path towards an csv file, reads the data into a weighted set.

    :param datafile: A string containing the physical path on the machine towards the dataset which the user desires
                     to use.
    :param problemType: A integer defining whether the dataset is used for regression or clustering.
    :return: A weighted set, namely, a PointSet object containing the dataset.
    """
    global PROBLEM_TYPE
    P = dataset
    # P = np.around(P, 6)  # round the dataset to avoid numerical instabilities
    # if 'lz' in PROBLEM_TYPE:  # if the problem is an instance of regression problem
    #     P[:, -1] = -P[:, -1]
    # else:
    #     P = preprocessData(P)
    return PointSet.PointSet(P=P, W=None, ellipsoid_max_iters=ELLIPSOID_MAX_ITER, problem_type=PROBLEM_TYPE,
                             use_svd=USE_SVD)


def checkIfFileExists(file_path):
    """
    The function at hand checks if a file at given path exists.

    :param file_path: A string which contains a path of file.
    :return: A boolean variable which counts for the existence of a file at a given path.
    """
    file = pathlib.Path(file_path)
    return file.exists()

def createDirectory(directory_name):
    """
    ##################### createDirectory ####################
    Input:
        - path: A string containing a path of an directory to be created at.

    Output:
        - None

    Description:
        This process is responsible creating an empty directory at a given path.
    """
    full_path = r'results/'+directory_name
    try:
        os.makedirs(full_path)
    except OSError:
        if not os.path.isdir(full_path):
            raise


def createRandomInitialVector(d):
    """
    This function create a random orthogonal matrix which each column can be use as an initial vector for
    regression problems.

    :param d: A scalar denoting a desired dimension.
    :return: None (using global we get A random orthogonal matrix).
    """
    global x0
    x0 = np.random.randn(d,d)  # random dxd matrix
    [x0, r] = np.linalg.qr(x0)  # attain an orthogonal matrix


############################################## Optimization methods ####################################################
def solveConvexRegressionProblem(P):
    start_time = time.time()
    w = cp.Variable(P.d, )

    loss = cp.sum(cp.multiply(P.W, cp.power(cp.abs(cp.matmul(P.P[:, :-1], w) - P.P[:, -1]), Z)))
    constraints = []

    prob = cp.Problem(cp.Minimize(loss), constraints)
    prob.solve()
    time_taken = time.time() - start_time

    print('Solving optimization problem in {:.4f} secs'.format(time_taken))
    return w.value, time_taken


def solveNonConvexRegressionProblem(P):
    start_time = time.time()
    func = lambda x: np.multiply(P.W, np.abs(np.dot(P[:, :-1], x) - P.P[:, -1]) ** Z)
    grad = lambda x: sp.optimize.approx_fprime(x, func, EPSILON)
    optimal_x = None
    optimal_val = np.Inf
    for i in range(OPTIMIZATION_NUM_INIT):
        x0 = createRandomInitialVector(P.d)
        res = sp.optimize.minimize(fun=func, x0=x0, jac=grad, method='L-BFGS-B')


def solveRegressionProblem(P):
    global PROBLEM
    if 'nc' in PROBLEM:
        return solveNonConvexRegressionProblem(P)
    else:
        return solveConvexRegressionProblem(P)


if __name__ == '__main__':
    # createSyntheticDan()
    # createRandomRegressionData()
    # testCauchy()
    # plotPointsBasedOnSens()
    # readRealData()
    pass
