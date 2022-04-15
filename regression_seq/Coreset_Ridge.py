import numpy as np
import copy
import random
global lamda
import  time
import warnings
import os
import scipy.io as scio
import pandas as pd
import MainProgram
import Utils
warnings.filterwarnings('ignore')

epsilon=1e-6
learning_rate=0.01

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)



def data_real(filename,lam=1):
    """
       deal the real dataset

       """
    global num, dim
    data = pd.read_csv(filename, sep=',', header=None)
    y = data.iloc[:, -1]
    X = data.iloc[:, :-1]
    num, dim = X.shape
    coreset_n=np.min([10000,0.02*num])
    parameters_set(coreset_n, lam)
    X = X.values
    c = np.ones([num, 1])
    X = np.hstack((X, c))
    y = y.values
    y = y.reshape(y.shape[0],1)
    return X,y


def parameters_set(n=1e4,lam=1000):
    """
    n is the coreset size
    """
    global coreset_size,lamd
    coreset_size=int(n)
    lamd=lam

def data_syn(N,d=20):
    """
       generate synthetic dataset

       """
    global num,dim,sigma
    num = int(N)
    dim=d
    sigma =2
    X=np.random.rand(num,dim)
    c=np.ones([num,1])
    X=np.hstack((X,c))
    h=(np.random.rand(dim+1,1)-0.5)*10
    y=np.dot(X,h)
    y=y+np.random.randn(num,1)*sigma
    print(np.mean(np.power(np.dot(X,h)-y,2))+10000*np.sum(np.power(h,2))/X.shape[0])
    return X,y,h



def coreset_lr(X, y, h, lamda):
    """
    layer sampling, and  the ratio of sample items preserve same for different layers .
    """
    f =np.power(np.dot(X, h) - y, 2) + np.ones((X.shape[0], 1)) * np.sum(lamda * np.power(h, 2), 0) / X.shape[0]
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


def FGD(max_iter, X,y,weight,h,lamda,lr=0.01):
    for iter in range(max_iter):
        gradient_f = (np.sum(weight*(2*(X*(np.dot(X, h) - y))),0)[:, np.newaxis]+2*lamda*h)/np.sum(weight)
        h -=lr*gradient_f
    return h

def Solution_Range(beta_title,beta,R):
    """
    Discriminate beta is out of the range of beta_title
    """
    if (np.sum(np.power(beta-beta_title,2)) >= R):
        return False
    else:
        return True





def lr_coreset_run(args,R_ball):
    """
    Our algorithm, when the beta reach the bound of beta_anc, reconstruct the coreset

    """
    X, y, lamda, epoch_max, init = args
    lr = learning_rate
    max_iter = 1
    index = np.random.permutation(X.shape[0])[0:coreset_size]
    weight_0=np.ones((coreset_size,1))*X.shape[0]/coreset_size
    beta_title=FGD(max_iter*10,X[index,:],y[index,:],weight_0, h=init,lamda=lamda, lr=lr)
    coreset_x,coreset_y,weight= coreset_lr(X,y,beta_title,lamda)
    beta=copy.deepcopy(beta_title)
    iteration = max_iter * epoch_max
    beta_title1=beta.copy()
    for iter in range(max_iter,max_iter*epoch_max,max_iter):
        beta = FGD(max_iter, coreset_x, coreset_y, weight, beta, lamda=lamda,lr=lr)
        R=R_ball
        if not Solution_Range(beta_title,beta,R):
            beta_title=copy.deepcopy(beta)
            coreset_x, coreset_y, weight = coreset_lr(X, y, beta_title,lamda)
        if np.linalg.norm(beta-beta_title1,2)<epsilon:
            print(iter)
            break
        beta_title1=beta.copy()
    return beta,iteration



def lr_run(args):
    """
    run the algorithm on original datasets to get initial beta
            """
    X, y,lamda,init,lr= args
    beta_title = FGD( 10, X, y, np.ones((X.shape[0], 1)),
                         h=init,lamda=lamda, lr=lr)
    return X, y, np.ones((X.shape[0], 1)),beta_title




def lr_coreset_once_run(args):
    """
  oneshot for  initial beta and coreset
    """
    X, y ,lamda,init,lr= args
    index = np.random.permutation(X.shape[0])[0:coreset_size]
    weight_0 = np.ones((coreset_size, 1)) * X.shape[0] / coreset_size
    beta_title = FGD(10, X[index, :], y[index, :], weight_0, h=init,lamda=lamda,
                                       lr=lr)
    coreset_x, coreset_y, weight = coreset_lr(X, y, beta_title,lamda)
    return  coreset_x, coreset_y, weight ,beta_title



def unisample_run(args):
    """
   uniform sample for  initial beta and coreset
    """
    X, y ,lamda,init,lr= args
    index = np.random.permutation(X.shape[0])[0:coreset_size]
    weight_0=np.ones((coreset_size,1))*X.shape[0]/coreset_size
    coreset_x,coreset_y,weight= X[index],y[index],weight_0
    beta_title = FGD(10, coreset_x, coreset_y, weight, init, lamda=lamda, lr=lr)
    return coreset_x,coreset_y,weight,beta_title



def near_convex_run(args):
    """
    importance sampling for  initial beta and coreset
    """
    X, y ,lamda,init,lr= args
    trainset = np.hstack((X, y))
    coreset = MainProgram.MainProgram.main(trainset.copy(), type='lz', sample_size=coreset_size,streaming=False)
    coreset_x,coreset_y,weight=coreset[0].P[:,0:dim+1],(coreset[0].P[:,dim+1])[:,np.newaxis],coreset[0].W[:,np.newaxis]
    beta_title = FGD(10, coreset_x, coreset_y, weight, init,lamda=lamda,  lr=lr)
    return coreset_x,coreset_y,weight,beta_title

def run(args):
    """
    Besides sequential methods，the same produces of other algorithms
    """
    X, y, lamda, epoch_max, init,sample_way = args
    lr = learning_rate
    max_iter = 1
    coreset_x, coreset_y, weight,beta_title = sample_way([X,y,lamda,init,lr])
    beta = copy.deepcopy(beta_title)
    iteration = max_iter * epoch_max
    for iter in range(max_iter, max_iter * epoch_max, max_iter):
        beta = FGD(max_iter, coreset_x, coreset_y, weight, beta,lamda=lamda,  lr=lr)
        if np.linalg.norm(beta - beta_title, 2) < epsilon:
            print('iter:', iter)
            iteration = iter
            break
        beta_title = beta.copy()
    return beta, iteration



def l2_loss_test(X_test, y_test, beta, lamda):
    loss = np.sum(np.power(np.dot(X_test, beta) - y_test, 2), 0) / X_test.shape[0]
    return np.float(loss)

def l2_loss_test_reg(X_test, y_test, beta, lamda):
    loss = np.sum(np.power(np.dot(X_test, beta) - y_test, 2), 0) / X_test.shape[0]+np.sum(lamda/9*np.power(beta,2),0)/X_test.shape[0]
    return np.float(loss)

def evaluate(X,y,X_test,y_test,beta,lamda):
    """
    Evaluate the beta result
    """
    alg_nums,_,_=np.shape(np.array(beta))
    beta_diff,loss,loss_train=[[] for _ in range(3)]
    for i in range(alg_nums):
        loss.append(l2_loss_test(X_test, y_test, beta[i], lamda=lamda))
        loss_train.append(l2_loss_test_reg(X_test, y_test, beta[i], lamda=lamda))
    scale = np.linalg.norm(beta[0], ord=2)
    for i in range(1,alg_nums):
        beta_diff.append(np.sum(np.linalg.norm(beta[0] - beta[i], ord=2) / scale).tolist())
    return beta_diff,loss,loss_train,scale

def main():
    lammingda, epoch_max = 0.01, 1000
    optimal, opt_path = FGD, 'FGD'
    n_range = [10,40,70,100, 400, 700, 1000]
    Num, dimension = 1e5, 50
    for d in [dimension]:
        X, y, h = data_syn(Num, d=d)
        train_size = int(0.9 * num)
        X_train, X_test = X[0:train_size, :], X[train_size + 1:num, :]
        y_train, y_test = y[0:train_size, :], y[train_size + 1:num, :]
        X, y = X_train, y_train
        Time, Time_var, Loss, Iter, Loss_var, Beta_var, Beta_diff, Predict, Predict_Var, Loss_train_avg, Loss_train_var, Scale = [
            [] for _ in range(12)]
        Alg = [lr_run,lr_coreset_once_run, unisample_run, near_convex_run, lr_coreset_run]
        Alg_name = ['Original','OneShot', 'UniSamp', 'ImpSamp', 'SeqCore-R']
        Init = (np.random.rand(dim + 1, 10) - 0.5) * 10
        iter_, beta_res_, timing_ = [], [], []
        for count_n, n in enumerate(n_range):
            lamda = lammingda * X.shape[0]
            parameters_set(n=n, lam=lamda)
            print(opt_path + '  coersetR: {}'.format(n))
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
                    if alg == lr_coreset_run:
                        parameters = [X, y, lamda, epoch_max, init]
                        R_ball = [0.5, 1,5, 10]
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
                beta_diff_, loss_, loss_train_, scale_ = evaluate(X, y, X_test, y_test, beta_res, lamda)
                beta_diff.append(beta_diff_)
                loss.append(loss_)
                loss_train.append(loss_train_)
                tttime.append(timing)
                if count_n == 0:
                    Scale.append(scale_)
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
        scio.savemat( 'coreset_syn_Ridge{}_ep{}_iter_{}_Samplemin_{}_N_{}_d_{}.mat'.format(learning_rate,epsilon,epoch_max, n_range[0],Num,dimension),
                     {'Time': np.array(Time), 'Loss_avg': np.array(Loss), 'Loss_var': np.array(Loss_var),
                      'Beta_diff': np.array(Beta_diff),
                       'Loss_train_avg': Loss_train_avg, 'Loss_train_var': Loss_train_var,
               })


if __name__=="__main__":
    main()
