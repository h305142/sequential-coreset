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
lamd=1
epsilon=1e-6
learning_rate=0.01

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)

set_seed(0)


def data_real(filename,lam=1):
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
    #n变为了coresetsize，Rd为几个R长度
    global coreset_size,lamd
    coreset_size=int(n)
    lamd=lam
    # return num,dim,coreset_size,sigma

def data_syn(d=20):
    global num,dim,sigma
    num = int(1e6)
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

def Cal_sample_base(coreset_size,N,k=1):
    All_size=0
    for i in range(N+1):
        All_size+=np.power(1+1/(np.power(2,i)*k),2)
    return np.floor(coreset_size/All_size)



def coreset_lr(X,y,h,lamda=lamd):
    f = np.power(np.dot(X, h) - y, 2) + np.ones((X.shape[0], 1)) * np.sum(lamda * np.power(h, 2), 0) / X.shape[0]
    X = np.hstack((X, y))
    L_max = np.max(f, 0)
    H = np.sum(f, 0) / X.shape[0]
    N = int(np.ceil(np.log2(L_max / H)))
    print(L_max, H, N)
    k = 1
    sample_base = Cal_sample_base(coreset_size=coreset_size, N=N, k=k)
    coreset = []
    Weight = []
    for i in range(1, N + 1):
        index_i = np.array(np.where((f > H * pow(2, i - 1)) & (f <= H * pow(2, i))))[0]
        sample_num_i = int(sample_base * np.power(1 + 1 / (np.power(2, i) * k), 2))
        if sample_num_i > 0:
            if len(coreset) == 0:
                if sample_num_i <= index_i.shape[0]:
                    choice = index_i[np.random.permutation(index_i.shape[0])[0:sample_num_i]]
                    coreset = X[choice, :]
                    Weight = np.ones((sample_num_i, 1)) * (index_i.shape[0] / sample_num_i)
                else:
                    coreset = X[index_i, :]
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
    if sample_num_i <= index_i.shape[0]:
        choice = index_i[np.random.permutation(index_i.shape[0])[0:sample_num_i]]
        coreset = np.vstack((coreset, X[choice, :]))
        Weight = np.vstack((Weight, np.ones((sample_num_i, 1)) * (index_i.shape[0] / sample_num_i)))
    else:
        coreset = np.vstack((coreset, X[index_i, :]))
        Weight = np.vstack((Weight, np.ones((len(index_i), 1))))

    return coreset[:, 0:dim + 1], coreset[:, dim + 1:dim + 2], Weight



def FGD(max_iter, X,y,weight,h,lamda=lamd,lr=0.01):
    for iter in range(max_iter):
        gradient_f = (np.sum(weight*(2*(X*(np.dot(X, h) - y))),0)[:, np.newaxis]+2*lamda*h)/np.sum(weight)
        h -=lr*gradient_f
    return h

def Solution_Range(beta_title,beta,R):
    if (np.sum(np.power(beta-beta_title,2)) >= R):
        # print('init {}'.format(np.sum(np.power(beta-beta_title,2))))
        return False
    else:
        return True


def l2_loss(X,y,beta,lamda=lamd):
    loss=np.sum(np.power(np.dot(X,beta)-y,2),0)/X.shape[0]+np.sum(lamda*np.power(beta,2),0)/X.shape[0]
    return np.float(loss)


def l2_loss_coreset(X,y,beta,weight,lamda=lamd):
    loss=np.sum(weight*(np.power(np.dot(X,beta)-y,2)),0)/np.sum(weight)+np.sum(lamda*np.power(beta,2),0)/np.sum(weight)
    return np.float(loss)

def lr_coreset_run(X,y,optimal,R_ball,lamda=lamd,epoch_max=1000):
    lr = learning_rate
    max_iter = 1
    index = np.random.permutation(X.shape[0])[0:coreset_size]
    weight_0=np.ones((coreset_size,1))*X.shape[0]/coreset_size
    beta_title=optimal(max_iter*10,X[index,:],y[index,:],weight_0, h=np.zeros((X.shape[1], 1)),lamda=lamda, lr=lr)
    coreset_x,coreset_y,weight= coreset_lr(X,y,beta_title)
    beta=copy.deepcopy(beta_title)
    l_before=l2_loss_coreset(coreset_x, coreset_y, beta,weight,lamda)
    iteration = []
    beta_title1=beta.copy()
    for iter in range(max_iter,max_iter*epoch_max,max_iter):
        beta = optimal(max_iter, coreset_x, coreset_y, weight, beta, lamda=lamda,lr=lr)
        R=R_ball
        if not Solution_Range(beta_title,beta,R):
            start=time.time()
            beta_title=copy.deepcopy(beta)
            coreset_x, coreset_y, weight = coreset_lr(X, y, beta_title)
        if (iter % 50 == 0):
            iteration.append(l2_loss(X, y, beta, lamda))
        if np.linalg.norm(beta-beta_title1,2)<epsilon:
            print(iter)
            break
        beta_title1=beta.copy()
    return beta,iteration


def lr_run(X,y,optimal,lamda=lamd,epoch_max=1000):
    lr=learning_rate
    max_iter = 1
    beta_title = optimal(max_iter*10, X, y, np.ones((X.shape[0], 1)),
                                   h=np.zeros((X.shape[1], 1)),lamda=lamda,lr=lr)
    l_before = l2_loss(X, y, beta_title,lamda)
    iteration =[]
    beta=beta_title.copy()
    for iter in range(max_iter,max_iter*epoch_max,max_iter):
        beta = optimal(max_iter, X, y, np.ones((X.shape[0], 1)), beta,lamda=lamda,
                                 lr=lr )
        if (iter % 50 == 0):
            iteration.append(l2_loss(X, y, beta, lamda))
        if np.linalg.norm(beta-beta_title,2)<epsilon:
            print(iter)
            break
        beta_title=beta.copy()
    return beta,iteration


def lr_coreset_once_run(X,y,optimal,lamda=lamd,epoch_max=1000):
    max_iter=1
    lr =learning_rate
    index = np.random.permutation(X.shape[0])[0:coreset_size]
    weight_0 = np.ones((coreset_size, 1)) * X.shape[0] / coreset_size
    beta_title = optimal(max_iter * 10, X[index, :], y[index, :], weight_0, h=np.zeros((X.shape[1], 1)),
                                       lamda=lamda, lr=lr)
    coreset_x, coreset_y, weight = coreset_lr(X, y, beta_title)
    beta = copy.deepcopy(beta_title)
    l_before = l2_loss_coreset(coreset_x, coreset_y, beta, weight, lamda)
    iteration =[]
    for iter in range(max_iter, max_iter * epoch_max, max_iter):
        beta = optimal(max_iter, coreset_x, coreset_y, weight, beta, lamda=lamda, lr=lr)
        if (iter % 50 == 0):
            iteration.append(l2_loss(X, y, beta, lamda))
        if np.linalg.norm(beta-beta_title,2)<epsilon:
            print(iter)
            break
        beta_title=beta.copy()
    return beta, iteration

def unisample_run(X,y,optimal,lamda=lamd,epoch_max=1000):
    lr=learning_rate
    max_iter = 1
    index = np.random.permutation(X.shape[0])[0:coreset_size]
    weight_0=np.ones((coreset_size,1))*X.shape[0]/coreset_size
    coreset_x,coreset_y,weight= X[index],y[index],weight_0
    beta_title = optimal(max_iter*10, coreset_x, coreset_y, weight, np.zeros((X.shape[1],1)), lamda=lamda, lr=lr)
    beta=copy.deepcopy(beta_title)
    iteration =[]
    l_before = l2_loss_coreset(coreset_x, coreset_y, beta,weight,lamda)
    for iter in range(max_iter,max_iter*epoch_max,max_iter):
        beta = optimal(max_iter, coreset_x, coreset_y, weight, beta, lamda=lamda,lr=lr)
        if (iter % 50 == 0):
            iteration.append(l2_loss(X, y, beta, lamda))
        if np.linalg.norm(beta-beta_title,2)<epsilon:
            print(iter)
            break
        beta_title=beta.copy()
    return beta,iteration

def near_convex_run(X,y,optimal,lamda=lamd,epoch_max=1000):
    lr=learning_rate
    max_iter = 1
    trainset = np.hstack((X, y))
    coreset = MainProgram.MainProgram.main(trainset.copy(), type='lz', sample_size=coreset_size,streaming=False)
    coreset_x,coreset_y,weight=coreset[0].P[:,0:dim+1],(coreset[0].P[:,dim+1])[:,np.newaxis],coreset[0].W[:,np.newaxis]
    beta_title = optimal(max_iter*10, coreset_x, coreset_y, weight, np.zeros((X.shape[1],1)), lamda=lamda, lr=lr)
    beta=copy.deepcopy(beta_title)
    iteration =[]
    l_before = l2_loss_coreset(coreset_x, coreset_y, beta,weight,lamda)
    for iter in range(max_iter,max_iter*epoch_max,max_iter):
        beta = optimal(max_iter, coreset_x, coreset_y, weight, beta, lamda=lamda,lr=lr)
        if (iter % 50 == 0):
            iteration.append(l2_loss(X, y, beta, lamda))
        if np.linalg.norm(beta-beta_title,2)<epsilon:
            print(iter)
            break
        beta_title=beta.copy()
    return beta,iteration

def l2_loss_test(X_test, y_test, beta, lamda=lamd):
    loss = np.sum(np.power(np.dot(X_test, beta) - y_test, 2), 0) / X_test.shape[0]
    return np.float(loss)



def main():
    lammingda=0.01
    print('lamda: ', lammingda)
    optimal = [FGD]
    opt_path=['FGD']
    opt=0
    n_range= [i for i in range(100, 1100, 100)]+[i for i in range(1000, 11000, 1000)]
    # n_range = np.hstack((n_range,[i for i in range(10000, 110000, 10000)]))
    for d in [50]:
        Time = []
        Loss = []
        Iter = []
        Loss_var = []
        Beta=[]
        Beta_diff=[]
        X, y, h = data_syn(d=d)
        train_size = int(0.9 * num)
        X_train, X_test = X[0:train_size, :], X[train_size + 1:num, :]
        y_train, y_test = y[0:train_size, :], y[train_size + 1:num, :]
        X, y = X_train, y_train
        time_1, flag = 0, 1
        beta_origin, iter1 = 0, 0
        Loss_train_avg,Loss_train_var=[],[]
        for n in n_range:
            lamda=lammingda*X.shape[0]
            parameters_set(n=n, lam=lamda)
            print(opt_path[opt] + '  coersetR: {}'.format(n))
            loss, tttime,loss_train = [], [],[]
            iter, beta_res = [], []
            beta_diff = []
            for i in range(10):
                timing = []
                print('times:', i)
                s = time.time()
                print('origin LR:')
                beta_origin, iter1 = lr_run(X, y, optimal=optimal[opt], lamda=lamda, epoch_max=1000)
                time_1 = time.time() - s
                timing.append(time_1)
                print('once coreset LR:')
                s = time.time()
                beta_layer, iter3 = lr_coreset_once_run(X, y, optimal=optimal[opt], lamda=lamda, epoch_max=1000)
                timing.append(time.time() - s)
                print('uniform LR:')
                s = time.time()
                beta_uniform, iter4 = unisample_run(X, y, optimal=optimal[opt], lamda=lamda, epoch_max=1000)
                timing.append(time.time() - s)
                print('sequential coreset2 LR:')
                s = time.time()
                beta_coreset2, iter5 = lr_coreset_run(X, y, optimal=optimal[opt], R_ball=0.5, lamda=lamda,
                                                      epoch_max=1000)
                timing.append(time.time() - s)
                print('sequential coreset3 LR:')
                s = time.time()
                beta_coreset3, iter6 = lr_coreset_run(X, y, optimal=optimal[opt], R_ball=1, lamda=lamda, epoch_max=1000)
                timing.append(time.time() - s)
                print('sequential coreset5 LR:')
                s = time.time()
                beta_coreset7, iter11 = lr_coreset_run(X, y, optimal=optimal[opt], R_ball=5, lamda=lamda,
                                                       epoch_max=1000)
                timing.append(time.time() - s)
                print('sequential coreset5 LR:')
                s = time.time()
                beta_coreset8, iter12 = lr_coreset_run(X, y, optimal=optimal[opt], R_ball=10, lamda=lamda,
                                                       epoch_max=1000)
                timing.append(time.time() - s)
                print('near convex')
                s = time.time()
                beta_near, iter7 = near_convex_run(X, y, optimal=optimal[opt], lamda=lamda, epoch_max=1000)
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
        path='result/test/Ridge/'
        if not os.path.exists(path):
            os.makedirs(path)
        scio.savemat(path+'coreset_Loss_lr001_cvge-6_{}.mat'.format(n_range[0]),{'Time':np.array(Time),'Loss_avg':np.array(Loss),'Loss_var':np.array(Loss_var),
                                                                        'Iter':np.array(Iter),'Beta_diff':np.array(Beta_diff),'Beta':np.array(Beta),'Loss_train_avg':np.array(Loss_train_avg),
                                                                        'Loss_train_var':np.array(Loss_train_var)})




if __name__=="__main__":
    main()
    # main2()
    # main()