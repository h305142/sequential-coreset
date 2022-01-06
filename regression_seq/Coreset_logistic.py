import numpy as np
import copy
import random
global lamda
import  time
import warnings
import os
import scipy.io as scio
import MainProgram
warnings.filterwarnings('ignore')
lamd=1
epsilon=1e-6
learning_rate=0.1
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)

set_seed()


def parameters_set(n=2e3,lam=1000,Rd=1):
    #n变为了coresetsize，Rd为几个R长度
    global coreset_size,lamd,R_ball
    coreset_size=int(n)
    lamd=lam
    R_ball=Rd
    # return num,dim,coreset_size,sigma

def data_syn(d=20):
    global num,dim,sigma
    num = int(1e6)
    dim=d
    sigma =1
    X = np.random.rand(num, dim)
    c = np.ones([num, 1])
    X = np.hstack((X, c))
    h = (np.random.rand(dim + 1, 1) - 0.5)*10
    y = np.dot(X, h)
    mean = np.mean(y)
    y = y  + np.random.randn(num, 1) * sigma
    y = (1 + np.sign(y - mean)) / 2
    h[dim] -= mean/2
    return X,y,h

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

def data_real(filename,lam=1):
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
    y = y.reshape(y.shape[0],1)
    y[y<0]=0
    return X,y


def Cal_sample_base(coreset_size,N,k=1):
    All_size=0
    for i in range(N+1):
        All_size+=np.power(1+1/(np.power(2,i)*k),2)
    return np.floor(coreset_size/All_size)

def coreset_lr(X,y,h,lamda=lamd):
    xbeta = np.dot(X, h)
    f = -y * xbeta + np.log(1 + np.exp(xbeta))
    X = np.hstack((X, y))
    L_max=np.max(f,0)
    H=np.sum(f,0)/X.shape[0]

    N=int(np.ceil(np.log2(L_max/H)))
    print(L_max, H, N)
    k=1
    sample_base=Cal_sample_base(coreset_size=coreset_size,N=N,k=k)
    coreset=[]
    Weight=[]
    for i in range(1,N+1):
        index_i=np.array(np.where((f>H*pow(2,i-1)) & (f<=H*pow(2,i))))[0]
        sample_num_i=int(sample_base*np.power(1+1/(np.power(2,i)*k),2))
        if sample_num_i>0:
            if len(coreset)==0:
                if sample_num_i <= index_i.shape[0]:
                    choice = index_i[np.random.permutation(index_i.shape[0])[0:sample_num_i]]
                    coreset =X[choice, :]
                    Weight = np.ones((sample_num_i, 1))*(index_i.shape[0] / sample_num_i)
                else:
                    coreset =  X[index_i, :]
                    Weight = np.ones((len(index_i), 1))
            else:
                if sample_num_i <= index_i.shape[0]:
                    choice = index_i[np.random.permutation(index_i.shape[0])[0:sample_num_i]]
                    coreset = np.vstack((coreset, X[choice, :]))
                    Weight = np.vstack((Weight, np.ones((sample_num_i, 1)) * (index_i.shape[0] / sample_num_i)))
                else:
                    coreset = np.vstack((coreset, X[index_i, :]))
                    Weight = np.vstack((Weight, np.ones((len(index_i), 1))))
    index_i = np.array(np.where(f <= H))[0]
    sample_num_i = coreset_size - coreset.shape[0]
    if sample_num_i<=index_i.shape[0]:
        choice = index_i[np.random.permutation(index_i.shape[0])[0:sample_num_i]]
        coreset = np.vstack((coreset, X[choice, :]))
        Weight = np.vstack((Weight, np.ones((sample_num_i, 1)) * (index_i.shape[0] / sample_num_i)))
    else:
        coreset = np.vstack((coreset, X[index_i, :]))
        Weight = np.vstack((Weight, np.ones((index_i.shape[0], 1))))
    return coreset[:,0:dim+1],coreset[:,dim+1:dim+2],Weight


def FGD(max_iter, X,y,weight,h,lr=0.01):
    """
        FGD优化regression
        """
    for iter in range(max_iter):
        xbeta = np.dot(X, h)
        gradient_f = np.sum(weight * (-y * X + (np.exp(xbeta) * X) / (1 + np.exp(xbeta))),0)[:, np.newaxis]/np.sum(weight)
        h -=lr*gradient_f
    return h


def Solution_Range(beta_title,beta,R):
    if (np.sum(np.power(beta-beta_title,2)) >= R):
        return False
    else:
        return True


def lg_coreset_run(X,y,optimal,R_ball,lamda=lamd,epoch_max=1000):
    """
        SGD优化sequential coreset regression的具体过程
        """
    max_iter = 1
    lr = learning_rate
    index = np.random.permutation(X.shape[0])[0:coreset_size]
    weight_0 = np.ones((coreset_size, 1)) * X.shape[0] / coreset_size
    beta_title=optimal(max_iter*10,X[index,:],y[index,:],weight_0, h=np.ones((X.shape[1], 1)), lr=lr)
    coreset_x,coreset_y,weight= coreset_lr(X,y,beta_title)
    beta=copy.deepcopy(beta_title)
    l_before = l2_loss_coreset(coreset_x, coreset_y, beta, weight)
    beta_title2=beta.copy()
    iteration = []
    for iter in range(max_iter,max_iter*epoch_max,max_iter):
        beta = optimal(max_iter, coreset_x, coreset_y, weight, beta,lr=lr)
        R=R_ball
        if not Solution_Range(beta_title,beta,R):
            start = time.time()
            beta_title=copy.deepcopy(beta)
            coreset_x, coreset_y, weight = coreset_lr(X, y, beta_title)
        if (iter % 50 == 0):
            iteration.append(l2_loss(X, y, beta, lamda))
        if np.linalg.norm(beta-beta_title2,2)<epsilon:
            print('iter:',iter)
            break
        beta_title2=beta.copy()
    return beta,iteration

def lg_run(X,y,optimal,lamda=lamd,epoch_max=1000):
    """
            SGD优化原始数据集regression的具体过程
            """
    max_iter = 1
    lr = learning_rate
    beta_title = optimal(max_iter*10, X, y, np.ones((X.shape[0], 1)),
                                   h=np.ones((X.shape[1], 1)),lr=lr)
    l_before = l2_loss(X, y, beta_title)
    iteration =[]
    beta=beta_title.copy()
    for iter in range(max_iter,max_iter*epoch_max,max_iter):
        beta = optimal(max_iter, X, y, np.ones((X.shape[0], 1)), beta,
                                 lr=lr )
        if (iter % 50 == 0):
            iteration.append(l2_loss(X, y, beta, lamda))
        if np.linalg.norm(beta-beta_title,2)<epsilon:
            print('iter:',iter)
            break
        beta_title=beta.copy()
    # print(beta_title)
    return beta,iteration

def lg_coreset_once_run(X,y,optimal,lamda=lamd,epoch_max=1000):
    max_iter = 1
    lr = learning_rate
    index = np.random.permutation(X.shape[0])[0:coreset_size]
    weight_0 = np.ones((coreset_size, 1)) * X.shape[0] / coreset_size
    beta_title = optimal(max_iter * 10, X[index, :], y[index, :], weight_0, h=np.ones((X.shape[1], 1)),
                                       lr=lr)
    coreset_x, coreset_y, weight = coreset_lr(X, y, beta_title)
    beta = copy.deepcopy(beta_title)
    l_before = l2_loss_coreset(coreset_x, coreset_y, beta, weight)
    iteration =[]
    for iter in range(max_iter, max_iter * epoch_max, max_iter):
        beta = optimal(max_iter, coreset_x, coreset_y, weight, beta,  lr=lr)
        if (iter % 50 == 0):
            iteration.append(l2_loss(X, y, beta, lamda))
        if np.linalg.norm(beta-beta_title,2)<epsilon:
            print('iter:', iter)
            break
        beta_title=beta.copy()
    # print(beta)
    return beta, iteration

def unisample_run(X,y,optimal,lamda=lamd,epoch_max=1000):
    max_iter = 1
    lr = learning_rate
    index = np.random.permutation(X.shape[0])[0:coreset_size]
    weight_0=np.ones((coreset_size,1))*X.shape[0]/coreset_size
    coreset_x,coreset_y,weight= X[index],y[index],weight_0
    beta_title = optimal(max_iter*10, coreset_x, coreset_y, weight, np.ones((X.shape[1],1)), lr=lr)
    beta=copy.deepcopy(beta_title)
    l_before = l2_loss_coreset(coreset_x, coreset_y, beta, weight)
    iteration = []
    for iter in range(max_iter,max_iter*epoch_max,max_iter):
        beta = optimal(max_iter, coreset_x, coreset_y, weight, beta,lr=lr)
        if (iter % 50 == 0):
            iteration.append(l2_loss(X, y, beta, lamda))
        if np.linalg.norm(beta-beta_title,2)<epsilon:
            print('iter:', iter)
            break
        beta_title=beta.copy()
    # print(beta)
    return beta,iteration




def near_convex_run(X,y,optimal,lamda=lamd,epoch_max=1000):
    lr=learning_rate
    max_iter = 1
    trainset = np.hstack((X, y))
    coreset = MainProgram.MainProgram.main(trainset.copy(), type='logistic', sample_size=coreset_size)
    coreset_x,coreset_y,weight=coreset[0].P[:,0:dim+1],(coreset[0].P[:,dim+1])[:,np.newaxis],coreset[0].W[:,np.newaxis]
    coreset_y[coreset_y<0]=0
    beta_title = optimal(max_iter*10, coreset_x, coreset_y, weight, np.ones((X.shape[1],1)), lr=lr)
    beta=copy.deepcopy(beta_title)
    iteration = []
    l_before = l2_loss_coreset(coreset_x, coreset_y, beta,weight)
    for iter in range(max_iter,max_iter*epoch_max,max_iter):
        beta = optimal(max_iter, coreset_x, coreset_y, weight, beta, lr=lr)
        if (iter % 50 == 0):
            iteration.append(l2_loss(X, y, beta, lamda))
        if np.linalg.norm(beta-beta_title,2)<epsilon:
            print('iter:', iter)
            break
        beta_title=beta.copy()
    return beta,iteration



def l2_loss_coreset(X,y,beta,weight):
    xbeta=np.dot(X, beta)
    loss=np.sum(weight*(-y*xbeta+np.log(1+np.exp(xbeta))),0)/sum(weight)

    return np.float(loss)

def l2_loss(X,y,beta, lamda=lamd):
    xbeta=np.dot(X, beta)
    loss=np.sum(-y*xbeta+np.log(1+np.exp(xbeta)),0)/X.shape[0]

    return np.float(loss)

def F_predict(X,y,beta):
    y_predict=np.dot(X,beta)
    y[y==0]=-1
    right=np.count_nonzero(y_predict*y>0)/y.shape[0]
    return np.float(right)

def l2_loss_test(X_test, y_test, beta, lamda=lamd):
    xbeta = np.dot(X_test, beta)
    loss = np.sum(-y_test * xbeta + np.log(1 + np.exp(xbeta)), 0) / X_test.shape[0]
    return np.float(loss)

def main():
    epoch_max=1000
    lamda = 10000
    print('lamda: ', lamda)
    optimal = [FGD]
    opt_path = ['FGD']
    n_range = [i for i in range(100, 1100, 100)] + [i for i in range(1000, 11000, 1000)]
    opt=0
    for d in [50]:
        X, y,h = data_syn(d=d)
        train_size = int(0.9 * num)
        X_train, X_test = X[0:train_size, :], X[train_size + 1:num, :]
        y_train, y_test = y[0:train_size, :], y[train_size + 1:num, :]
        X, y = X_train, y_train
        Time = []
        Loss = []
        Iter = []
        Loss_var = []
        Beta = []
        Beta_diff = []
        Predict=[]
        Loss_train_avg=[]
        Loss_train_var=[]
        time_1, flag = 0, 1
        beta_origin, iter1 = 0, 0
        for n in n_range:
            parameters_set(n=n, lam=lamda)
            print(opt_path[opt] + '  coersetR: {}'.format(n))
            loss, tttime,loss_train = [], [],[]
            iter, beta_res = [], []
            beta_diff = []
            predict=[]
            alg_nums=8
            for i in range(10):
                timing= []
                print('times:', i)
                s = time.time()
                print('origin LR:')
                beta_origin, iter1 = lg_run(X, y, optimal=optimal[opt], lamda=lamda, epoch_max=epoch_max)
                time_1 = time.time() - s
                timing.append(time_1)
                print('once coreset LR:')
                s = time.time()
                beta_layer, iter3 = lg_coreset_once_run(X, y, optimal=optimal[opt], lamda=lamda, epoch_max=epoch_max)
                timing.append(time.time() - s)
                print('uniform LR:')
                s = time.time()
                beta_uniform, iter4 = unisample_run(X, y, optimal=optimal[opt], lamda=lamda, epoch_max=epoch_max)
                timing.append(time.time() - s)
                print('sequential coreset2 LR:')
                s = time.time()
                beta_coreset2, iter5=lg_coreset_run(X, y, optimal=optimal[opt], R_ball=0.5,lamda=lamda, epoch_max=epoch_max)
                timing.append(time.time() - s)
                print('sequential coreset3 LR:')
                s = time.time()
                beta_coreset3, iter6 = lg_coreset_run(X, y, optimal=optimal[opt], R_ball=1, lamda=lamda, epoch_max=epoch_max)
                timing.append(time.time() - s)
                print('sequential coreset5 LR:')
                s = time.time()
                beta_coreset7, iter11 = lg_coreset_run(X, y, optimal=optimal[opt], R_ball=5, lamda=lamda,
                                                       epoch_max=epoch_max)
                timing.append(time.time() - s)
                print('sequential coreset5 LR:')
                s = time.time()
                beta_coreset8, iter12 = lg_coreset_run(X, y, optimal=optimal[opt], R_ball=10, lamda=lamda,
                                                       epoch_max=epoch_max)
                timing.append(time.time() - s)
                print('near convex')
                s = time.time()
                beta_near, iter7 = near_convex_run(X, y, optimal=optimal[opt], lamda=lamda, epoch_max=epoch_max)
                timing.append(time.time() - s)
                tttime.append(timing)
                iter.append([iter1, iter3, iter4, iter5, iter6, iter11, iter12,iter7])
                beta_res.append(
                    [beta_origin, beta_layer, beta_uniform, beta_coreset2, beta_coreset3,
                     beta_coreset7,beta_coreset8,beta_near])
                scale = np.linalg.norm(beta_origin, ord=2)
                beta_diff.append([np.sum(np.linalg.norm(beta_origin - beta_layer,ord=2) / scale).tolist(),
                                  np.sum(np.linalg.norm(beta_origin - beta_uniform,ord=2) / scale).tolist(),
                                  np.sum(np.linalg.norm(beta_origin - beta_coreset2,ord=2) / scale).tolist(),
                                  np.sum(np.linalg.norm(beta_origin - beta_coreset3,ord=2) / scale).tolist(),

                                  np.sum(np.linalg.norm(beta_origin - beta_coreset7, ord=2) / scale).tolist(),
                                  np.sum(np.linalg.norm(beta_origin - beta_coreset8, ord=2) / scale).tolist(),
                                  np.sum(np.linalg.norm(beta_origin - beta_near,ord=2) / scale).tolist()])
                loss.append([l2_loss_test(X_test, y_test, beta_origin, lamda=lamda),l2_loss_test(X_test, y_test, beta_layer, lamda=lamda),l2_loss_test(X_test, y_test, beta_uniform, lamda=lamda),
                             l2_loss_test(X_test, y_test, beta_coreset2, lamda=lamda),
                             l2_loss_test(X_test, y_test, beta_coreset3, lamda=lamda),
                              l2_loss_test(X_test, y_test, beta_coreset7, lamda=lamda),
                              l2_loss_test(X_test, y_test, beta_coreset8, lamda=lamda),
                              l2_loss_test(X_test,y_test,beta_near,lamda=lamda)])
                loss_train.append([l2_loss(X, y, beta_origin, lamda=lamda),
                             l2_loss(X, y, beta_layer, lamda=lamda),
                             l2_loss(X, y, beta_uniform, lamda=lamda),
                             l2_loss(X, y, beta_coreset2, lamda=lamda),
                             l2_loss(X, y, beta_coreset3, lamda=lamda),
                             l2_loss(X, y, beta_coreset7, lamda=lamda),
                             l2_loss(X, y, beta_coreset8, lamda=lamda),
                             l2_loss(X, y, beta_near, lamda=lamda)])
                predict1 = [F_predict(X_test, y_test.copy(), beta_origin),
                            F_predict(X_test, y_test.copy(), beta_layer),
                            F_predict(X_test, y_test.copy(), beta_uniform),
                            F_predict(X_test, y_test.copy(), beta_coreset2),
                            F_predict(X_test, y_test.copy(), beta_coreset3),
                            F_predict(X_test, y_test.copy(), beta_coreset7),
                            F_predict(X_test, y_test.copy(), beta_coreset8),
                            F_predict(X_test, y_test.copy(), beta_near)
                            ]
                predict.append(predict1)
            Predict.append(np.mean(predict,0).tolist())
            Iter = np.mean(iter, 0)
            Loss.append(np.mean(loss, 0).tolist())
            Loss_var.append(np.std(loss, 0).tolist())
            Loss_train_avg.append(np.mean(loss_train, 0).tolist())
            Loss_train_var.append(np.std(loss_train, 0).tolist())
            Time.append(np.mean(tttime, 0).tolist())
            Beta_diff.append(np.mean(beta_diff, 0).tolist())
            Beta.append(beta_res)
            print('loss: ', Loss)
            print('time: ', Time)
            print('beta_diff: ', Beta_diff)
            print('Acc',Predict)
        path = 'result/test/syn'
        if not os.path.exists(path):
            os.makedirs(path)
        scio.savemat(path + 'coreset_Loss_lr001_cvg1e-6_{}.mat'.format(n_range[0]),
                     {'Time': np.array(Time), 'Loss_avg': np.array(Loss), 'Loss_var': np.array(Loss_var),
                      'Iter': np.array(Iter), 'Beta_diff': np.array(Beta_diff), 'Beta': np.array(Beta),
                      'Accuracy': np.array(Predict),'Loss_train_avg':Loss_train_avg,'Loss_train_var':Loss_train_var})



if __name__=="__main__":
    main()
    # main()