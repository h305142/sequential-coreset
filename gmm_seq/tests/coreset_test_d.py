from __future__ import division
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
# %matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.io as scio
import time
import sys
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.datasets import make_blobs, make_classification
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
np.random.seed(42)
from utils import datagen2, plotting
import coresets
import algorithms
import random

# log
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def kc_eval(true_label, pred_label, z):  # true_label,pred_label
    N = len(true_label)
    # cluster centers
    class_label = np.array(list(set(true_label)))
    cluster_label = np.array(list(set(pred_label)))
    true_label = np.array(true_label)
    pred_label = np.array(pred_label)
    count = 0

    for i in range(0, cluster_label.shape[0]):
        member_idx = np.where(pred_label == cluster_label[i])
        member_true_label = true_label[member_idx]
        frequency = np.bincount(member_true_label).max()
        count = count + frequency

    purity = count / (N);
    return purity


Timer_avg_table = []
Loss_avg_table = []
Purity_avg_table = []
Timer_std_table = []
Loss_std_table = []
Purity_std_table = []
# help('utils')

# set parameters
sys.stdout = Logger("./log/coreset_test_d.txt")

# algorithmic detailed parameters
init_params = 'kmeans'
reg_covar = 1e-6
tol = 1e-6
max_iter = 1000
max_iter_seq = max_iter

# datasets parameters:
# n: datasize, k: number of components, coreset_*: assign the coreset size
n = 100000
d_set = [4,8,12,16,20,24]
k_set = [4,8,12,16,20,24]
coreset_proportion = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]

d = 12 #
n_components = 12
coreset_size = int(2000)

rpts = 20

for d in d_set:
    #true_k = n_components
    #coreset_size = int(n * coreset_ratio)
    print("d=",d,"n_components=",n_components,"-----------------------------------------")

    loss = []
    timer = []
    purity = []

    for i in range(rpts):
        X, target_label = datagen2.gen_crp(
            n=n, d=d, n_components=n_components, method=1, cluster_std=120, random_state=None)
        # full
        print('----repeat round:', i, ',full dataset:')
        wgm = algorithms.WeightedGaussianMixture(n_components=n_components, max_iter=max_iter, tol=tol, \
                                                 reg_covar=reg_covar, init_params=init_params)
        start = time.time()
        wgm.fit(X, np.ones(X.shape[0]))
        end = time.time()
        duration_full = end - start
        assignment = wgm.predict(X)
        loss_fulldata_full = -wgm.score(X, y=None)
        loss_coreset_full = wgm.cal_f1_loss(X)
        print("---Negative Logarithmic Likelihood(in full data)---=", loss_fulldata_full,loss_coreset_full)
        print('y labels:', set(assignment))
        purity_full = kc_eval(target_label, assignment, 0)
        print("time = ", duration_full, "purity = ", purity_full)


        ## uniform
        print('----repeat round:', i, ',uniform sampling:')
        uniform_gen = coresets.KMeansUniformCoreset(X)
        C_u, w_u = uniform_gen.generate_coreset(coreset_size)
        wgm = algorithms.WeightedGaussianMixture(n_components=n_components, max_iter=max_iter, tol=tol, \
                                                   reg_covar=reg_covar, init_params=init_params)
        start = time.time()
        wgm.fit(C_u,w_u)
        end = time.time()
        duration_uni = end - start
        assignment = wgm.predict(X)
        loss_fulldata_uni = - wgm.score(X, y=None)
        loss_coreset_uni = wgm.cal_f1_loss(C_u)
        print("---Negative Logarithmic Likelihood(in full data)---=", loss_fulldata_uni)
        print("---Negative Logarithmic Likelihood(in weighted coreset)---=", loss_coreset_uni)
        print('y labels:', set(assignment))
        purity_uni = kc_eval(target_label, assignment, 0)
        print("time = ", duration_uni, "purity = ", purity_uni)

        ##coreset
        print('----repeat round:', i, ',coreset:')
        coreset_gen = coresets.KMeansCoreset(X, n_clusters=n_components)
        C, w = coreset_gen.generate_coreset(coreset_size)
        wgm = algorithms.WeightedGaussianMixture(n_components=n_components, max_iter=max_iter, tol=tol, \
                                                   reg_covar=reg_covar, init_params=init_params)
        start = time.time()
        wgm.fit(C, w)
        end = time.time()
        duration_coreset = end - start
        assignment = wgm.predict(X)
        loss_fulldata_imp = -wgm.score(X, y=None)
        loss_coreset_imp = wgm.cal_f1_loss(C)
        print("---Negative Logarithmic Likelihood(in full data)---=", loss_fulldata_imp)
        print("---Negative Logarithmic Likelihood(in weighted coreset)---=", loss_coreset_imp)
        print('y labels:', set(assignment))
        purity_coreset = kc_eval(target_label, assignment, 0)
        print("time = ", duration_coreset, "purity_coreset = ", purity_coreset)

        ## seq_core
        print('----repeat round:', i, ',seq_core:')
        wgm = algorithms.WeightedGaussianMixture(n_components=n_components, max_iter=max_iter_seq, tol=tol, reg_covar=reg_covar,\
                                                   init_params=init_params, seq_coreset=True, coreset_size = coreset_size)
        start = time.time()
        wgm.fit(X, np.ones(X.shape[0]))
        end = time.time()
        duration_seq = end - start
        assignment = wgm.predict(X)
        loss_fulldata_seq = - wgm.score(X, y=None)
        print("---Negative Logarithmic Likelihood(in full data)---=", loss_fulldata_seq)
        print('y labels:', set(assignment))
        purity_seq = kc_eval(target_label, assignment, 0)
        print("time = ", duration_seq, "purity_seq = ", purity_seq)

        ## summary
        loss.append([loss_fulldata_full,loss_fulldata_uni,loss_fulldata_imp,loss_fulldata_seq])
        timer.append([duration_full,duration_uni,duration_coreset,duration_seq])
        purity.append([purity_full,purity_uni,purity_coreset,purity_seq])
        # print("times:",timer)
        # print("loss:",loss)
        # print("purity:",purity)
    loss_avg = np.mean(loss,0).tolist()
    timer_avg = np.mean(timer,0).tolist()
    purity_avg = np.mean(purity,0).tolist()
    loss_std = np.std(loss, 0).tolist()
    timer_std = np.std(timer, 0).tolist()
    purity_std = np.std(purity, 0).tolist()

    print("timer_avg:", timer_avg)
    print("loss_avg:", loss_avg)
    print("purity_avg:", purity_avg)
    print("timer_std:", timer_std)
    print("loss_std:", loss_std)
    print("purity_std:", purity_std)
    Timer_avg_table.append(timer_avg)
    Loss_avg_table.append(loss_avg)
    Purity_avg_table.append(purity_avg)
    Timer_std_table.append(timer_std)
    Loss_std_table.append(loss_std)
    Purity_std_table.append(purity_std)

print("Timer_avg_table = ", Timer_avg_table)
print("Loss_avg_table = ", Loss_avg_table)
print("Purity_avg_table = ", Purity_avg_table)
print("Timer_std_table = ", Timer_std_table)
print("Loss_std_table = ", Loss_std_table)
print("Purity_std_table = ", Purity_std_table)
path = 'result/'
if not os.path.exists(path):
    os.mkdir(path)
scio.savemat(path + 'gmm_{}.mat'.format('d'),
             {'Timer_avg_table': np.array(Timer_avg_table), 'Loss_avg_table': np.array(Loss_avg_table), \
              'Purity_avg_table': np.array(Purity_avg_table), 'Timer_std_table': np.array(Timer_std_table),
              'Loss_std_table': np.array(Loss_std_table), 'Purity_std_table': np.array(Purity_std_table)})


