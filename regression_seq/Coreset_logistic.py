import numpy as np
import copy
import random

global lamda
import time
import warnings
import os
import scipy.io as scio
import MainProgram
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')
epsilon = 1e-3
learning_rate = 1


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)



def parameters_set(n=2e3, lam=1000, Rd=1):
    """
     n is the coreset size
     """
    global coreset_size, lamd, R_ball
    coreset_size = int(n)
    lamd = lam
    R_ball = Rd


def data_syn(N,d=20):
    """
    generate synthetic dataset

    """
    global num, dim, sigma
    num = int(N)
    dim = d
    sigma = 1
    X = np.random.rand(num, dim)
    c = np.ones([num, 1])
    X = np.hstack((X, c))
    h = (np.random.rand(dim + 1, 1) - 0.5) * 10
    y = np.dot(X, h)
    mean = np.mean(y)
    y = y + np.random.randn(num, 1) * sigma

    y = (1 + np.sign(y - mean)) / 2
    h[dim] -= mean / 2
    return X, y, h





def data_real(filename, lam=1):
    """
    deal the real dataset

    """
    global num, dim
    data = pd.read_csv(filename, sep=',', header=None)
    y = data.iloc[:, -1]
    X = data.iloc[:, :-1]

    X = X.values
    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    num, dim = X.shape
    c = np.ones([num, 1])
    X = np.hstack((X, c))
    y = y.values
    y = y.reshape(y.shape[0], 1)
    y[y < 0] = 0
    return X, y


def coreset_lr(X, y, h):
    """
          layer sampling, and  the ratio of sample items preserve same for different layers .
          """
    xbeta = np.dot(X, h)
    f = -y * xbeta + np.log(1 + np.exp(xbeta))
    X = np.hstack((X, y))
    L_max = np.max(f, 0)
    H = np.sum(f, 0) / X.shape[0]

    N = int(np.ceil(np.log2(L_max / H)))
    print(L_max, H, N)
    sample_base = coreset_size / X.shape[0]
    coreset = []
    Weight = []
    for i in range(1, N + 1):
        index_i = np.array(np.where((f > H * pow(2, i - 1)) & (f <= H * pow(2, i))))[0]
        sample_num_i = int(sample_base * index_i.shape[0])
        if sample_num_i > 0:
            choice = index_i[np.random.permutation(index_i.shape[0])[0:sample_num_i]]
            if len(coreset) == 0:
                coreset = X[choice, :]
                Weight = np.ones((sample_num_i, 1)) * (index_i.shape[0] / sample_num_i)
            else:
                coreset = np.vstack((coreset, X[choice, :]))
                Weight = np.vstack((Weight, np.ones((sample_num_i, 1)) * (index_i.shape[0] / sample_num_i)))
    index_i = np.array(np.where(f <= H))[0]
    sample_num_i = coreset_size - len(coreset)
    choice = index_i[np.random.permutation(index_i.shape[0])[0:sample_num_i]]
    if len(coreset) == 0:
        coreset = X[choice, :]
        Weight = np.ones((sample_num_i, 1)) * (index_i.shape[0] / sample_num_i)
    else:
        coreset = np.vstack((coreset, X[choice, :]))
        Weight = np.vstack((Weight, np.ones((sample_num_i, 1)) * (index_i.shape[0] / sample_num_i)))

    return coreset[:, 0:dim + 1], coreset[:, dim + 1:dim + 2], Weight


def FGD(max_iter, X, y, weight, h, lr=0.01):
    for iter in range(max_iter):
        xbeta = np.dot(X, h)
        gradient_f = np.sum(weight * (-y * X + (np.exp(xbeta) * X) / (1 + np.exp(xbeta))), 0)[:, np.newaxis] / np.sum(
            weight)
        h -= lr * gradient_f
    return h


def Solution_Range(beta_title, beta, R):
    """
    Discriminate beta is out of the range of beta_title
    """
    if (np.sum(np.power(beta - beta_title, 2)) >= R):
        return False
    else:
        return True


def lg_coreset_run(args, R_ball):
    """
         Our algorithm, when the beta reach the bound of beta_anc, reconstruct the coreset

         """
    X, y, lamda, epoch_max, init = args
    max_iter = 1
    lr = learning_rate
    index = np.random.permutation(X.shape[0])[0:coreset_size]
    weight_0 = np.ones((coreset_size, 1)) * X.shape[0] / coreset_size
    beta_title = FGD(max_iter * 10, X[index, :], y[index, :], weight_0, h=init, lr=lr)
    coreset_x, coreset_y, weight = coreset_lr(X, y, beta_title)
    beta = copy.deepcopy(beta_title)
    beta_title2 = beta.copy()
    iteration = max_iter * epoch_max
    for iter in range(max_iter, max_iter * epoch_max, max_iter):
        beta = FGD(max_iter, coreset_x, coreset_y, weight, beta, lr=lr)
        R = R_ball
        if not Solution_Range(beta_title, beta, R):
            beta_title = copy.deepcopy(beta)
            coreset_x, coreset_y, weight = coreset_lr(X, y, beta_title)
        if np.linalg.norm(beta - beta_title2, 2) < epsilon:
            print('iter:', iter)
            iteration = iter
            break
        beta_title2 = beta.copy()

    return beta, iteration


def lg_run(args):
    """
       run the algorithm on original datasets to get initial beta
               """
    X, y, init, lr = args
    beta_title = FGD(10, X, y, np.ones((X.shape[0], 1)),
                     h=init, lr=lr)
    return X, y, np.ones((X.shape[0], 1)), beta_title


def lg_coreset_once_run(args):
    """
    oneshot for  initial beta and coreset
      """
    X, y, init, lr = args
    index = np.random.permutation(X.shape[0])[0:coreset_size]
    weight_0 = np.ones((coreset_size, 1)) * X.shape[0] / coreset_size
    beta_title = FGD(10, X[index, :], y[index, :], weight_0, h=init,
                     lr=lr)
    coreset_x, coreset_y, weight = coreset_lr(X, y, beta_title)

    return coreset_x, coreset_y, weight, beta_title


def unisample_run(args):
    """
         uniform sample for  initial beta and coreset
          """
    X, y, init, lr = args
    index = np.random.permutation(X.shape[0])[0:coreset_size]
    weight_0 = np.ones((coreset_size, 1)) * X.shape[0] / coreset_size
    coreset_x, coreset_y, weight = X[index], y[index], weight_0
    beta_title = FGD(10, coreset_x, coreset_y, weight, init, lr=lr)
    return coreset_x, coreset_y, weight, beta_title


def near_convex_run(args):
    """
       importance sampling for  initial beta and coreset
       """
    X, y, init, lr = args
    trainset = np.hstack((X, y))
    coreset = MainProgram.MainProgram.main(trainset.copy(), type='logistic', sample_size=coreset_size)
    coreset_x, coreset_y, weight = coreset[0].P[:, 0:dim + 1], (coreset[0].P[:, dim + 1])[:, np.newaxis], coreset[0].W[
                                                                                                          :, np.newaxis]
    coreset_y[coreset_y < 0] = 0
    beta_title = FGD(10, coreset_x, coreset_y, weight, init, lr=lr)
    return coreset_x, coreset_y, weight, beta_title


def run(args):
    """
      Besides sequential methods，the same produces of other algorithms
      """
    X, y, lamda, epoch_max, init, sample_way = args
    lr = learning_rate
    max_iter = 1
    coreset_x, coreset_y, weight, beta_title = sample_way([X, y, init, lr])
    beta = copy.deepcopy(beta_title)
    iteration = max_iter * epoch_max
    for iter in range(max_iter, max_iter * epoch_max, max_iter):
        beta = FGD(max_iter, coreset_x, coreset_y, weight, beta, lr=lr)
        if np.linalg.norm(beta - beta_title, 2) < epsilon:
            print('iter:', iter)
            iteration = iter
            break
        beta_title = beta.copy()
    return beta, iteration


def l2_loss(X, y, beta, lamda):
    xbeta = np.dot(X, beta)
    loss = np.sum(-y * xbeta + np.log(1 + np.exp(xbeta)), 0) / X.shape[0]

    return np.float(loss)


def F_predict(X, y, beta):
    y_predict = np.dot(X, beta)
    y[y == 0] = -1
    right = np.count_nonzero(y_predict * y > 0) / y.shape[0]
    return np.float(right)


def l2_loss_test(X_test, y_test, beta, lamda):
    xbeta = np.dot(X_test, beta)
    loss = np.sum(-y_test * xbeta + np.log(1 + np.exp(xbeta)), 0) / X_test.shape[0]
    return np.float(loss)


def evaluate(X, y, X_test, y_test, beta, lamda):
    """
      Evaluate the beta result
      """
    alg_nums, _, _ = np.shape(np.array(beta))
    beta_diff, loss, loss_train, predict = [[] for _ in range(4)]
    for i in range(alg_nums):
        loss.append(l2_loss_test(X_test, y_test, beta[i], lamda=lamda))
        loss_train.append(l2_loss(X, y, beta[i], lamda=lamda))
        predict.append(F_predict(X_test, y_test.copy(), beta[i]))
    scale = np.linalg.norm(beta[0], ord=2)
    for i in range(1, alg_nums):
        beta_diff.append(np.sum(np.linalg.norm(beta[0] - beta[i], ord=2) / scale).tolist())
    return beta_diff, loss, loss_train, predict, scale


def main():
    epoch_max = 1000
    lamda = 10000
    opt_path = 'FGD'
    n_range= [10,40,70,100,400,700,1000]
    Num,dimension=1e5,50
    for d in [dimension]:
        X, y, h = data_syn(Num,d=d)
        index_new = np.random.permutation(X.shape[0])
        train_index = index_new[0:int(0.9 * X.shape[0])]
        test_index = index_new[int(0.9 * X.shape[0] + 1):X.shape[0]]
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index, :], y[test_index, :]
        X, y = X_train, y_train
        Time, Time_var, Loss, Iter, Loss_var, Beta_var, Beta_diff, Predict, Predict_Var, Loss_train_avg, Loss_train_var, Scale = [
            [] for _ in range(12)]
        Alg = [lg_run, lg_coreset_once_run, unisample_run, near_convex_run, lg_coreset_run]
        Alg_name = ['Original', 'OneShot', 'UniSamp', 'ImpSamp', 'SeqCore-R']
        Init = (np.random.rand(dim + 1, 10) - 0.5) * 10
        iter_, beta_res_, timing_ = [], [], []
        for count_n, n in enumerate(n_range):
            parameters_set(n=n, lam=lamda)
            print(opt_path + '  coersetR: {}'.format(n))
            Itering = []
            loss, loss_train, predict, beta_diff, tttime = [[] for _ in range(5)]
            for i in range(10):
                iter, beta_res, timing = [], [], []
                if count_n != 0:
                    iter.append(iter_[i])
                    beta_res.append(beta_res_[i])
                    timing.append(timing_[i])
                print('times:', i)
                for alg_num, alg in enumerate(Alg):
                    init = copy.deepcopy(Init[:, i][:, np.newaxis])
                    if alg_num == 0 and count_n != 0:
                        continue
                    print(Alg_name[alg_num])
                    if alg == lg_coreset_run:
                        parameters = [X, y, lamda, epoch_max, init]
                        R_ball = [0.5, 1, 5, 10]
                        for rball in R_ball:
                            time_now = time.time()
                            print('R_ball={}'.format(rball))
                            beta, itering = alg(parameters, R_ball=rball)
                            beta_res.append(beta)
                            iter.append(itering)
                            timing.append(time.time() - time_now)
                    else:
                        parameters = [X, y, lamda, epoch_max, init, alg]
                        time_now = time.time()
                        beta, itering = run(parameters)
                        beta_res.append(beta)
                        iter.append(itering)
                        timing.append(time.time() - time_now)
                        if alg_num == 0 and count_n == 0:
                            iter_.append(iter[0])
                            beta_res_.append(beta_res[0])
                            timing_.append(timing[0])
                Itering.append(iter)
                beta_diff_, loss_, loss_train_, predict_, scale_ = evaluate(X, y, X_test, y_test, beta_res, lamda)
                beta_diff.append(beta_diff_)
                loss.append(loss_)
                loss_train.append(loss_train_)
                predict.append(predict_)
                tttime.append(timing)
                if count_n == 0:
                    Scale.append(scale_)
            Iter.append(np.mean(Itering, 0).tolist())
            Predict.append(np.mean(predict, 0).tolist())
            Predict_Var.append(np.std(predict, 0).tolist())
            Loss.append(np.mean(loss, 0).tolist())
            Loss_var.append(np.std(loss, 0).tolist())
            Loss_train_avg.append(np.mean(loss_train, 0).tolist())
            Loss_train_var.append(np.std(loss_train, 0).tolist())
            Time.append(np.mean(tttime, 0).tolist())
            Time_var.append(np.std(tttime, 0).tolist())
            Beta_diff.append(np.mean(beta_diff, 0).tolist())
            Beta_var.append(np.std(beta_diff, 0).tolist())
            print('loss: ', Loss)
            print('time: ', Time)
            print('beta_diff: ', Beta_diff)
            print('Acc', Predict)
        scio.savemat('coreset_syn_logistic{}_ep{}_iter_{}_Samplemin_{}_N_{}_d_{}.mat'.format(learning_rate,epsilon,epoch_max, n_range[0],Num,dimension),
                     {'Time': np.array(Time), 'Loss_avg': np.array(Loss), 'Loss_var': np.array(Loss_var),
                      'Iter': np.array(Iter), 'Beta_diff': np.array(Beta_diff),
                      'Accuracy': np.array(Predict),'Loss_train_avg':Loss_train_avg,'Loss_train_var':Loss_train_var,
                     'Time_var':np.array(Time_var),'Beta_var':np.array(Beta_var),'Predict_var':np.array(Predict_Var)})


if __name__ == "__main__":
    main()
