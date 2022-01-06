import numpy as np
import copy
import random
global lamda
from pac_sklearn.linear_model import _ridge
import  time
import warnings
import os
import scipy.io as scio
import pandas as pd
from Booster import *
import MainProgram
warnings.filterwarnings('ignore')
lamd=1
epsilon=0.001

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)

set_seed(0)


# def data_real(filename,lam=1):
#
#     data = pd.read_csv(filename, sep=',', header=None)
#     y = data.iloc[:, -1]
#     X = data.iloc[:, :-1]
#     num, d = X.shape
#     parameters_set(num, d, lam)
#     X = X.values
#     c = np.ones([num, 1])
#     X = np.hstack((X, c))
#     y = y.values
#     y = y.reshape(y.shape[0],1)
#     return X,y
# filename = 'E:/work/JupyterLab/dataPre/kc_house_data.csv/house_price.csv'
# X,y = data_real(filename, lam=lamda)

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
    print(np.mean(np.power(np.dot(X,h)-y,2))+1000*np.sum(np.power(h,2))/X.shape[0])
    return X,y,h

def Cal_sample_base(coreset_size,N,k=1):
    All_size=0
    for i in range(N+1):
        All_size+=np.power(1+1/(np.power(2,i)*k),2)
    return np.floor(coreset_size/All_size)



def coreset_lr(X,y,h,lamda=lamd):
    f=np.power(np.dot(X,h)-y,2)+np.ones((X.shape[0],1))*np.sum(lamda*np.power(h,2),0)/num
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
        Weight = np.vstack((Weight, np.ones((len(index_i), 1)) ))

    return coreset[:,0:dim+1],coreset[:,dim+1:dim+2],Weight


def SGD(max_iter, X,y,weight,h,lamda=lamd,lr=0.01):
    """
    weight should cover regulartion,also regulartion needs to consider into each fi
    for each lamda in other functions, I add the divde num(/num) for them
    """
    for iter in range(max_iter):
        sample_index=np.random.randint(0,X.shape[0],1)
        gradient_f = weight[sample_index]*(2*(X[sample_index]*(np.dot(X[sample_index], h) - y[sample_index])).T+2*lamda*h/num)
        h -=lr*gradient_f
    return h

def FGD(max_iter, X,y,weight,h,lamda=lamd,lr=0.01):
    for iter in range(max_iter):
        gradient_f = (np.sum(weight*(2*(X*(np.dot(X, h) - y))),0)[:, np.newaxis]+2*lamda*h)/num
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
    # coreset_build_time=0
    # coreset_num=0
    lr1 = 0.1
    lr2 = 0.1
    max_iter = 1
    index = np.random.permutation(X.shape[0])[0:coreset_size]
    weight_0=np.ones((coreset_size,1))*X.shape[0]/coreset_size
    beta_title=optimal(max_iter*10,X[index,:],y[index,:],weight_0, h=np.zeros((X.shape[1], 1)),lamda=lamda, lr=lr1)
    coreset_x,coreset_y,weight= coreset_lr(X,y,beta_title)
    beta=copy.deepcopy(beta_title)
    l_before = l2_loss(X, y, beta,lamda)
    # l_before=l2_loss_coreset(coreset_x, coreset_y, beta,weight,lamda)
    iteration = max_iter * epoch_max
    for iter in range(max_iter,max_iter*epoch_max,max_iter):
        beta = optimal(max_iter, coreset_x, coreset_y, weight, beta, lamda=lamda,lr=lr2)
        R=R_ball
        if not Solution_Range(beta_title,beta,R):
            start=time.time()
            beta_title=copy.deepcopy(beta)
            coreset_x, coreset_y, weight = coreset_lr(X, y, beta_title)
            # coreset_build_time+=(time.time()-start)
            # coreset_num+=1
        l_now =l2_loss(X, y, beta,lamda)
        # l_now = l2_loss_coreset(coreset_x, coreset_y, beta, weight, lamda)
        if (np.abs(l_now - l_before) < epsilon):
            print('iter :', iter)
            iteration=iter
            break
        l_before = l_now
    return beta,iteration


def lr_run(X,y,optimal,lamda=lamd,epoch_max=1000):
    lr=0.1
    max_iter = 1
    beta_title = optimal(max_iter*10, X, y, np.ones((X.shape[0], 1)),
                                   h=np.zeros((X.shape[1], 1)),lamda=lamda,lr=lr)
    l_before = l2_loss(X, y, beta_title,lamda)
    iteration = max_iter * epoch_max
    for iter in range(max_iter,max_iter*epoch_max,max_iter):
        beta_title = optimal(max_iter, X, y, np.ones((X.shape[0], 1)), beta_title,lamda=lamda,
                                 lr=lr )
        l_now = l2_loss(X, y, beta_title,lamda)
        if (np.abs(l_now - l_before) < epsilon):
            print('iter :', iter)
            iteration=iter
            break
        l_before = l_now
    return beta_title,iteration


def lr_coreset_once_run(X,y,optimal,lamda=lamd,epoch_max=1000):
    max_iter=1
    lr1 = 0.1
    lr2 = 0.1
    index = np.random.permutation(X.shape[0])[0:coreset_size]
    weight_0 = np.ones((coreset_size, 1)) * X.shape[0] / coreset_size
    beta_title = optimal(max_iter * 10, X[index, :], y[index, :], weight_0, h=np.zeros((X.shape[1], 1)),
                                       lamda=lamda, lr=lr1)
    coreset_x, coreset_y, weight = coreset_lr(X, y, beta_title)
    beta = copy.deepcopy(beta_title)
    l_before = l2_loss(X, y, beta,lamda)
    # l_before = l2_loss_coreset(coreset_x, coreset_y, beta, weight, lamda)
    iteration = max_iter * epoch_max
    for iter in range(max_iter, max_iter * epoch_max, max_iter):
        beta = optimal(max_iter, coreset_x, coreset_y, weight, beta, lamda=lamda, lr=lr2)
        l_now =l2_loss(X, y, beta,lamda)
        # l_now=l2_loss_coreset(coreset_x, coreset_y, beta,weight,lamda)
        if (np.abs(l_now - l_before) < epsilon):
            print('iter :', iter)
            iteration = iter
            break
        l_before = l_now
    return beta, iteration

def unisample_run(X,y,optimal,lamda=lamd,epoch_max=1000):
    lr=0.1
    max_iter = 1
    index = np.random.permutation(X.shape[0])[0:coreset_size]
    weight_0=np.ones((coreset_size,1))*X.shape[0]/coreset_size
    coreset_x,coreset_y,weight= X[index],y[index],weight_0
    beta_title = optimal(max_iter*10, coreset_x, coreset_y, weight, np.zeros((X.shape[1],1)), lamda=lamda, lr=lr)
    beta=copy.deepcopy(beta_title)
    l_before = l2_loss(X, y, beta,lamda)
    iteration = max_iter * epoch_max
    # l_before = l2_loss_coreset(coreset_x, coreset_y, beta,weight,lamda)
    for iter in range(max_iter,max_iter*epoch_max,max_iter):
        beta = optimal(max_iter, coreset_x, coreset_y, weight, beta, lamda=lamda,lr=lr)
        l_now =l2_loss(X, y, beta,lamda)
        # l_now=l2_loss_coreset(coreset_x, coreset_y, beta,weight,lamda)
        if (np.abs(l_now - l_before) < epsilon):
            print('iter :', iter)
            iteration=iter
            break
        l_before = l_now
    return beta,iteration

def LMS_run(X,y,optimal,lamda=lamd,epoch_max=1000):
    max_iter=1
    lr1 = 0.1
    lr2 = 0.1
    coreset_x, coreset_y, weight=get_coresets_LMS(X.copy(), y.copy(), np.ones((X.shape[0],1)), folds=3)
    beta_title = optimal(max_iter * 10, coreset_x, coreset_y,weight,
                         h=np.zeros((coreset_x.shape[1], 1)), lamda=lamda, lr=lr1)
    l_before = l2_loss(X, y, beta_title, lamda)
    iteration = max_iter * epoch_max
    for iter in range(max_iter, max_iter * epoch_max, max_iter):
        beta_title = optimal(max_iter, coreset_x, coreset_y,weight, beta_title, lamda=lamda,
                             lr=lr2)
        l_now = l2_loss(X, y, beta_title, lamda)
        if (np.abs(l_now - l_before) < epsilon):
            print('iter :', iter)
            iteration = iter
            break
        l_before = l_now
    return beta_title,iteration

def near_convex_run(X,y,optimal,lamda=lamd,epoch_max=1000):
    lr=1e-1
    max_iter = 1
    trainset = np.hstack((X, y))
    coreset = MainProgram.MainProgram.main(trainset.copy(), type='lz', sample_size=coreset_size)
    coreset_x,coreset_y,weight=coreset[0].P[:,0:dim+1],(coreset[0].P[:,dim+1])[:,np.newaxis],coreset[0].W[:,np.newaxis]
    beta_title = optimal(max_iter*10, coreset_x, coreset_y, weight, np.zeros((X.shape[1],1)), lamda=lamda, lr=lr)
    beta=copy.deepcopy(beta_title)
    l_before = l2_loss(X, y, beta,lamda)
    iteration = max_iter * epoch_max
    # l_before = l2_loss_coreset(coreset_x, coreset_y, beta,weight,lamda)
    for iter in range(max_iter,max_iter*epoch_max,max_iter):
        beta = optimal(max_iter, coreset_x, coreset_y, weight, beta, lamda=lamda,lr=lr)
        l_now =l2_loss(X, y, beta,lamda)
        # l_now=l2_loss_coreset(coreset_x, coreset_y, beta,weight,lamda)
        if (np.abs(l_now - l_before) < epsilon):
            print('iter :', iter)
            iteration=iter
            break
        l_before = l_now
    return beta,iteration


def main():
    lamda=1000
    print('lamda: ', lamda)
    optimal = [FGD]
    opt_path=['FGD']
    opt=0
    # n_range=[i for i in range(1000,11000,1000)]
    # n_range = np.hstack((n_range,[i for i in range(10000, 110000, 10000)]))
    for d in range(5,16):
        Time = []
        Loss = []
        Iter = []
        Loss_var = []
        Beta = []
        Beta_diff = []
        for n in [1e4]:
            X, y, h = data_syn(d=d)
            parameters_set(n=n, lam=lamda)
            print(opt_path[opt] + '  coersetR: {}'.format(n))
            loss, tttime = [], []
            iter, beta_res = [], []
            beta_diff = []
            time_1, flag = 0, 1
            beta_origin, iter1 = 0, 0
            for i in range(10):
                timing = []
                print('times:', i)
                s = time.time()
                if flag == 1:
                    print('origin LR:')
                    beta_origin, iter1 = lr_run(X, y, optimal=optimal[opt], lamda=lamda, epoch_max=1000)
                    time_1 = time.time() - s
                    timing.append(time_1)
                    flag = 2
                else:
                    timing.append(time_1)
                print('once coreset LR:')
                s = time.time()
                beta_layer, iter3 = lr_coreset_once_run(X, y, optimal=optimal[opt], lamda=lamda, epoch_max=1000)
                timing.append(time.time() - s)
                print('uniform LR:')
                s = time.time()
                beta_uniform, iter4 = unisample_run(X, y, optimal=optimal[opt], lamda=lamda, epoch_max=1000)
                timing.append(time.time() - s)
                print('sequential coreset1 LR:')
                s = time.time()
                beta_coreset1, iter2 = lr_coreset_run(X, y, optimal=optimal[opt], R_ball=0.1, lamda=lamda,
                                                      epoch_max=1000)
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
                print('Near Convex:')
                s = time.time()
                beta_near, iter7 = near_convex_run(X, y, optimal=optimal[opt], lamda=lamda, epoch_max=1000)
                timing.append(time.time() - s)
                print('LMS LR:')
                s = time.time()
                beta_LMS, iter8 = LMS_run(X, y, optimal=optimal[opt], lamda=lamda, epoch_max=1000)
                tttime.append(timing)
                iter.append([iter1, iter3, iter4, iter2, iter5, iter6, iter7, iter8])
                beta_res.append(
                    [beta_origin, beta_layer, beta_uniform, beta_coreset1, beta_coreset2, beta_coreset3, beta_near,
                     beta_LMS])
                scale = np.linalg.norm(beta_origin, ord=2)
                beta_diff.append([np.sum(np.linalg.norm(beta_origin - beta_layer, ord=2) / scale).tolist(),
                                  np.sum(np.linalg.norm(beta_origin - beta_uniform, ord=2) / scale).tolist(),
                                  np.sum(np.linalg.norm(beta_origin - beta_coreset1, ord=2) / scale).tolist(),
                                  np.sum(np.linalg.norm(beta_origin - beta_coreset2, ord=2) / scale).tolist(),
                                  np.sum(np.linalg.norm(beta_origin - beta_coreset3, ord=2) / scale).tolist(),
                                  np.sum(np.linalg.norm(beta_origin - beta_near, ord=2) / scale ).tolist(),
                                  np.sum(np.linalg.norm(beta_origin - beta_LMS, ord=2) / scale ).tolist()])
                loss.append([l2_loss(X, y, beta_origin, lamda=lamda), l2_loss(X, y, beta_coreset1, lamda=lamda),
                             l2_loss(X, y, beta_layer, lamda=lamda), l2_loss(X, y, beta_uniform, lamda=lamda),
                             l2_loss(X, y, beta_coreset2, lamda=lamda), l2_loss(X, y, beta_coreset3, lamda=lamda),
                             l2_loss(X, y, beta_near, lamda=lamda), l2_loss(X, y, beta_LMS, lamda=lamda)])
            Iter = np.mean(iter, 0)
            Loss.append(np.mean(loss, 0).tolist())
            Loss_var.append(np.var(loss, 0).tolist())
            Time.append(np.mean(tttime, 0).tolist())
            Beta_diff.append(np.mean(beta_diff, 0).tolist())
            Beta.append(beta_res)
            print('loss: ', Loss)
            print('time: ', Time)
            print('beta_diff: ', Beta_diff)
        path = 'result/Ridge/'
        if not os.path.exists(path):
            os.mkdir(path)
        scio.savemat(path + 'coreset_2_d_{}_n_1e4.mat'.format(opt_path[opt]),
                     {'Time': np.array(Time), 'Loss_avg': np.array(Loss), 'Loss_var': np.array(Loss_var),
                      'Iter': np.array(Iter), 'Beta_diff': np.array(Beta_diff), 'Beta': np.array(Beta)})



if __name__=="__main__":
    main()
    # main2()
    # main()