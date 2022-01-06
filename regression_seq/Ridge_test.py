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

def parameters_set(n=2e4,lam=1000,Rd=1):
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
    sigma = 1
    X=np.random.rand(num,dim)
    c=np.ones([num,1])
    X=np.hstack((X,c))
    h=(np.random.rand(dim+1,1)-0.5)*10

    # index_max=np.random.permutation(h.shape[0])[0:int(0.2*dim)]
    # h[index_max,:]=h[index_max,:]*10
    # index_min = np.random.permutation(h.shape[0])[0:int(0.2 * dim)]
    # h[index_min, :] = h[index_min,:] *0.1
    y=np.dot(X,h)
    y=y+np.random.randn(num,1)*sigma
    print(np.max(y), np.min(y))
    noise_index=np.random.permutation(y.shape[0])[0:int(0.05 * num)]
    y[noise_index] = y[noise_index] + np.ones((int(0.05 * num), 1)) *(np.max(y)-np.min(y))
    print(np.max(y),np.min(y))
    print(h)
    return X,y

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
    # sample_ratio=coreset_size/X.shape[0]
    coreset=[]
    Weight=[]
    # ss=[]
    # index_i = np.array(np.where(f <= H))[0]
    # ss.append(len(index_i))
    # s=[]
    for i in range(1,N+1):
        index_i=np.array(np.where((f>H*pow(2,i-1)) & (f<=H*pow(2,i))))[0]
        sample_num_i=int(sample_base*np.power(1+1/(np.power(2,i)*k),2))
        # ss.append(len(index_i))
        if sample_num_i>0:
            # choice = index_i[np.random.permutation(index_i.shape[0])[0:sample_num_i]]
            # s.append(sample_num_i)
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
    # print('each layer:',ss)
    index_i = np.array(np.where(f <= H))[0]
    sample_num_i = coreset_size - coreset.shape[0]
    # s.append(sample_num_i)
    # print('sample:',s)
    if sample_num_i<=index_i.shape[0]:
        choice = index_i[np.random.permutation(index_i.shape[0])[0:sample_num_i]]
        coreset = np.vstack((coreset, X[choice, :]))
        Weight = np.vstack((Weight, np.ones((sample_num_i, 1)) * (index_i.shape[0] / sample_num_i)))
    else:
        coreset = np.vstack((coreset, X[index_i, :]))
        Weight = np.vstack((Weight, np.ones((len(index_i), 1)) ))

    return coreset[:,0:dim+1],coreset[:,dim+1:dim+2],Weight

# def coreset_lr(X,y,h,lamda=lamd):
#     f=np.power(np.dot(X,h)-y,2)+np.ones((X.shape[0],1))*np.sum(lamda*np.power(h,2),0)/num
#     X = np.hstack((X, y))
#     L_max=np.max(f,0)
#     H=np.sum(f,0)/X.shape[0]
#     N=int(np.ceil(np.log2(L_max/H)))
#     print(L_max,H,N)
#     sample_ratio=coreset_size/X.shape[0]
#     coreset=[]
#     Weight=[]
#     index_i = np.array(np.where(f <= H))[0]
#     s=[]
#     s.append(len(index_i))
#     for i in range(1,N+1):
#         index_i=np.array(np.where((f>H*pow(2,i-1)) & (f<=H*pow(2,i))))[0]
#         sample_num_i=int(np.floor(index_i.shape[0]*sample_ratio))
#         s.append(len(index_i))
#         if sample_num_i>0:
#             choice = index_i[np.random.permutation(index_i.shape[0])[0:sample_num_i]]
#             if len(coreset)==0:
#                 coreset = X[choice, :]
#                 Weight =  np.ones((sample_num_i, 1))*(index_i.shape[0] / sample_num_i)
#             else:
#                 coreset=np.vstack((coreset,X[choice,:]))
#                 Weight=np.vstack((Weight,np.ones((sample_num_i,1))*(index_i.shape[0]/sample_num_i)))
#     print(s)
#     index_i = np.array(np.where(f <= H))[0]
#     sample_num_i = coreset_size - coreset.shape[0]
#
#     if sample_num_i<=index_i.shape[0]:
#         choice = index_i[np.random.permutation(index_i.shape[0])[0:sample_num_i]]
#         coreset = np.vstack((coreset, X[choice, :]))
#         Weight = np.vstack((Weight, np.ones((sample_num_i, 1)) * (index_i.shape[0] / sample_num_i)))
#     else:
#         coreset = np.vstack((coreset, X[index_i, :]))
#         Weight = np.vstack((Weight, np.ones((index_i.shape[0], 1)) ))
#     return coreset[:,0:dim+1],coreset[:,dim+1:dim+2],Weight


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

def Solution_Range(beta_title,beta,X,y,R,lamda=lamd):
    F_init=np.power(np.dot(X,beta_title)-y,2)+np.ones((X.shape[0],1))*np.sum(lamda*np.power(beta_title,2),0)/num
    F=np.power(np.dot(X,beta)-y,2)+np.ones((X.shape[0],1))*np.sum(lamda*np.power(beta,2),0)/num
    # if (np.max(np.abs(F_init-F),0)>=R):
    if ((np.mean(np.abs(F_init - F))) >= R):
        return False
    else:
        return True


def lr_coreset_run(X,y,optimal,lamda=lamd,epoch_max=1000):
    coreset_build_time=0
    coreset_num=0
    if optimal==SGD:
        lr1=0.001
        lr2=0.0001
        epoch_max = 10000
        max_iter = 1000
    else:
        lr1 = 0.1
        lr2 = 0.1
        max_iter = 1
    index = np.random.permutation(X.shape[0])[0:coreset_size]
    weight_0=np.ones((coreset_size,1))*X.shape[0]/coreset_size
    beta_title=optimal(max_iter*10,X[index,:],y[index,:],weight_0, h=np.zeros((X.shape[1], 1)),lamda=lamda, lr=lr1)
    coreset_x,coreset_y,weight= coreset_lr(X,y,beta_title)
    beta=copy.deepcopy(beta_title)
    l_before = l2_loss(X, y, beta,lamda)
    iteration = max_iter * epoch_max
    for iter in range(max_iter,max_iter*epoch_max,max_iter):
        beta = optimal(max_iter, coreset_x, coreset_y, weight, beta, lamda=lamda,lr=lr2)
        R=R_ball*(np.sum(weight*(np.power(np.dot(coreset_x,beta)-coreset_y,2)))+np.sum(lamda*np.power(beta,2),0))/num
        if not Solution_Range(beta_title,beta,coreset_x,coreset_y,R):
            start=time.time()
            beta_title=copy.deepcopy(beta)
            coreset_x, coreset_y, weight = coreset_lr(X, y, beta_title)
            coreset_build_time+=(time.time()-start)
            coreset_num+=1
        l_now =l2_loss(X, y, beta,lamda)
        if (np.abs(l_now - l_before) < epsilon):
            print('iter :', iter)
            iteration=iter
            break
        l_before = l_now.copy()
    print(beta)
    return beta,iteration,coreset_build_time,coreset_num


def lr_run(X,y,optimal,lamda=lamd,epoch_max=1000):
    if optimal==SGD:
        lr=0.01
        epoch_max = 10000
        max_iter = 1000
    else:
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
        l_before = l_now.copy()
    return beta_title,iteration

def lr_LMS_run(X,y,optimal,lamda=lamd,epoch_max=1000):
    # if optimal==SGD:
    #     epoch_max=10000
    #     max_iter=1000
    #     lr1=0.001
    #     lr2=0.0001
    # else:
    #     max_iter=1
    #     lr1 = 0.1
    #     lr2 = 0.1
    # coreset_x, coreset_y, weight=get_coresets_LMS(X, y, np.ones((X.shape[0],1)), folds=3)
    # beta_title = optimal(max_iter * 10, coreset_x, coreset_y,weight,
    #                      h=np.zeros((coreset_x.shape[1], 1)), lamda=lamda, lr=lr1)
    # l_before = l2_loss(X, y, beta_title, lamda)
    # iteration = max_iter * epoch_max
    # for iter in range(max_iter, max_iter * epoch_max, max_iter):
    #     beta_title = optimal(max_iter, coreset_x, coreset_y,weight, beta_title, lamda=lamda,
    #                          lr=lr2)
    #     l_now = l2_loss(X, y, beta_title, lamda)
    #     if (np.abs(l_now - l_before) < epsilon):
    #         print('iter :', iter)
    #         iteration = iter
    #         break
    #     l_before = l_now.copy()
    beta_title=coreset_train_LMS(X,y, np.ones((X.shape[0],1)), lamda, folds=3, solver='ridge')
    return beta_title[:, np.newaxis], 0

def lr_coreset_once_run(X,y,optimal,lamda=lamd,epoch_max=1000):
    if optimal==SGD:
        epoch_max=10000
        max_iter=1000
        lr1=0.001
        lr2=0.0001
    else:
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
    iteration = max_iter * epoch_max
    for iter in range(max_iter, max_iter * epoch_max, max_iter):
        beta = optimal(max_iter, coreset_x, coreset_y, weight, beta, lamda=lamda, lr=lr2)
        l_now =l2_loss(X, y, beta,lamda)
        if (np.abs(l_now - l_before) < epsilon):
            print('iter :', iter)
            iteration = iter
            break
        l_before = l_now.copy()
    return beta, iteration

def unisample_run(X,y,optimal,lamda=lamd,epoch_max=1000):
    if optimal == SGD:
        epoch_max=10000
        max_iter=1000
        lr1=0.001
        lr2=0.0001
    else:
        max_iter=1
        lr1 = 0.1
        lr2 = 0.1
    coreset_x, coreset_y, weight=get_coresets_LMS(X, y, np.ones((X.shape[0],1)), folds=3)
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
        l_before = l_now.copy()
    return beta_title,iteration

def unisample_uniform_run(X,y,optimal,lamda=lamd,epoch_max=1000):
    if optimal==SGD:
        lr=0.0001
        epoch_max = 10000
        max_iter = 1000
    else:
        lr=0.1
        max_iter = 1
    index = np.random.permutation(X.shape[0])[0:coreset_size]
    weight_0=np.ones((coreset_size,1))*X.shape[0]/coreset_size
    coreset_x,coreset_y,weight= X[index],y[index],weight_0
    beta_title = optimal(max_iter*10, coreset_x, coreset_y, weight, np.zeros((X.shape[1],1)), lamda=lamda, lr=lr)
    beta=copy.deepcopy(beta_title)
    l_before = l2_loss(X, y, beta,lamda)
    iteration = max_iter * epoch_max
    for iter in range(max_iter,max_iter*epoch_max,max_iter):
        beta = optimal(max_iter, coreset_x, coreset_y, weight, beta, lamda=lamda,lr=lr)
        # R = R_ball * (np.sum(weight * (np.power(np.dot(coreset_x, beta) - coreset_y, 2))) + np.sum(
        #     lamda * np.power(beta, 2), 0)) / num
        # if not Solution_Range(beta_title, beta, coreset_x, coreset_y, R):
        #     beta_title = copy.deepcopy(beta)
        #     index = np.random.permutation(X.shape[0])[0:coreset_size]
        #     weight_0 = np.ones((coreset_size, 1)) * X.shape[0] / coreset_size
        #     coreset_x, coreset_y, weight = X[index], y[index], weight_0
        l_now =l2_loss(X, y, beta,lamda)
        if (np.abs(l_now - l_before) < epsilon):
            print('iter :', iter)
            iteration=iter
            break
        l_before = l_now.copy()
    return beta,iteration



def l2_loss(X,y,beta,lamda=lamd):
    loss=np.sum(np.power(np.dot(X,beta)-y,2),0)/X.shape[0]+np.sum(lamda*np.power(beta,2),0)/X.shape[0]

    return loss

def main():
    lamda=1000
    print('lamda: ', lamda)
    optimal = [FGD,SGD]
    opt_path=['FGD','SGD']
    X, y = data_syn(d=20)
    for opt in range(1):
        Time = []
        Loss = []
        Iter = []
        Loss_var = []
        Coreset_time=[]
        Coreset_number=[]
        for n in [20000]:
            parameters_set(n=n,lam=lamda,Rd=0.1)
            flag = 1
            print(opt_path[opt]+'  coersetR: {}'.format(1))
            loss_origin_FGD,loss_coreset_FGD,loss_origin_SGD,loss_coreset_SGD,loss_coreset_LMS=[],[],[],[],[]
            start0,end0,end1,end2,end3,end4,end5,end6=[],[],[],[],[],[],[],[]
            iter1,iter2,iter3,iter4,iter5=[],[],[],[],[]
            coreset_build_time=[]
            coreset_build_number=[]
            #Iter只是最后一次结果，非平均的，FGD，SGD写在optimal里面
            for i in range(5):
                print('times:', i)
                if(optimal[opt]==SGD or flag==2):
                        start0 .append( time.time())
                        print('origin LR:')
                        beta_origin_FGD, iter = lr_run(X, y, optimal=optimal[opt], lamda=lamda, epoch_max=200)
                        flag=2
                        end0.append(time.time())
                        loss_origin_FGD.append(l2_loss(X, y, beta_origin_FGD, lamda=lamda))
                        iter1.append(iter)
                end1.append(time.time())
                print('sequential coreset LR:')
                beta_coreset_FGD,iter,timing,number =lr_coreset_run(X, y, optimal=optimal[opt], lamda=lamda, epoch_max=200)
                iter2.append(iter)
                coreset_build_time.append(timing)
                coreset_build_number.append(number)
                end2.append(time.time())
                print('once coreset LR:')
                beta_origin_SGD,iter = lr_coreset_once_run(X, y, optimal=optimal[opt], lamda=lamda, epoch_max=200)
                iter3.append(iter)
                end3 .append(time.time())
                # print('LMS solver:')
                # beta_coreset_SGD,iter =unisample_run(X, y, optimal=optimal[opt], lamda=lamda, epoch_max=200)
                # iter4.append(iter)
                end4.append(time.time())

                # print('LMS GD:')
                # beta_coreset_LMS, iter = lr_LMS_run(X, y, optimal=optimal[opt], lamda=lamda, epoch_max=200)
                # iter5.append(iter)
                end5.append(time.time())
                print('uniform GD:')
                beta_unform,iter=unisample_uniform_run(X, y, optimal=optimal[opt], lamda=lamda, epoch_max=200)
                end6.append(time.time())
                loss_uniform=l2_loss(X, y, beta_unform, lamda=lamda)
                print('uniform Loss:',loss_uniform)
                loss_coreset_FGD.append(l2_loss(X,y,beta_coreset_FGD,lamda=lamda))
                loss_origin_SGD.append(l2_loss(X, y, beta_origin_SGD, lamda=lamda))
                # loss_coreset_SGD.append( l2_loss(X, y, beta_coreset_SGD, lamda=lamda))
                # loss_coreset_LMS.append(l2_loss(X, y, beta_coreset_LMS, lamda=lamda))
            loss_origin_FGD, loss_coreset_FGD, loss_origin_SGD, loss_coreset_SGD,var1,var2,var3,var4 = np.mean(loss_origin_FGD), np.mean(
                    loss_coreset_FGD), np.mean(loss_origin_SGD), np.mean(loss_coreset_SGD),np.var(loss_origin_FGD),np.var(loss_coreset_FGD),\
                np.var(loss_origin_SGD),np.var(loss_coreset_SGD)
            time1,time2,time3,time4,time5=np.mean(np.array(end0)-np.array(start0)),np.mean(np.array(end2)-np.array(end1)),np.mean(np.array(end3)-np.array(end2))\
            , np.mean(np.array(end4) - np.array(end3)),np.mean(np.array(end5) - np.array(end4))
            iter1,iter2,iter3,iter4=np.mean(iter1),np.mean(iter2),np.mean(iter3),np.mean(iter4),
            Coreset_time.append(np.mean(coreset_build_time))
            Coreset_number.append(np.mean(coreset_build_number))
            if len(Time)==0:
                Time=[[time1],[time2],[time3],[time4]]
                Loss=[[loss_origin_FGD],[loss_coreset_FGD],[loss_origin_SGD],[loss_coreset_SGD]]
                Loss_var=[[var1],[var2],[var3],[var4]]
                Iter=[[iter1],[iter2],[iter3],[iter4]]

            else:
                Time=np.hstack((Time,[[time1],[time2],[time3],[time4]]))
                Loss = np.hstack((Loss, [[loss_origin_FGD],[loss_coreset_FGD],[loss_origin_SGD],[loss_coreset_SGD]]))
                Iter=np.hstack((Iter,[[iter1],[iter2],[iter3],[iter4]]))
                Loss_var = np.hstack((Loss_var,[[var1], [var2], [var3], [var4]]))
            print('loss: loss_origin ',loss_origin_FGD,' loss_coreset ', loss_coreset_FGD ,' once coreset: ', loss_origin_SGD,' LMS GD: ',loss_coreset_SGD,
                  'LMS solver',np.mean(loss_coreset_LMS))
            print('origin time:{},  coreset time{} ,once: {}, uniform: {}, LMS:{}'.format(time1,time2,time3,time4,time5),'  uniform loss:',np.mean(np.array(end6)- np.array(time5)))
        Time=np.array(Time)
        path='result/Ridge/'
        if not os.path.exists(path):
            os.mkdir(path)
        scio.savemat(path+'coreset_test2_{}.mat'.format(opt_path[opt]),{'Time':np.array(Time),'Loss_avg':np.array(Loss),'Loss_var':np.array(Loss_var),
                                                                        'Iter':np.array(Iter),'Coreset_build_time':np.array(Coreset_time),
                                                                        'Coreset_build_number':Coreset_number})

# def main2():
#     lamda=1000
#     print('lamda: ', lamda)
#     optimal = [FGD,SGD]
#     opt_path=['FGD','SGD']
#     X, y = data_syn(d=20)
#     for opt in range(2):
#         Time = []
#         Loss = []
#         Iter = []
#         Loss_var = []
#         Coreset_time=[]
#         Coreset_number=[]
#         for rd in range(1,6):
#             parameters_set(n=2e4,lam=lamda,Rd=rd)
#             flag = 1
#             print(opt_path[opt]+'  coersetR: {}'.format(1))
#             loss_origin_FGD,loss_coreset_FGD,loss_origin_SGD,loss_coreset_SGD=[],[],[],[]
#             start0,end0,end1,end2,end3,end4=[],[],[],[],[],[]
#             iter1,iter2,iter3,iter4=[],[],[],[]
#             coreset_build_time=[]
#             coreset_build_number=[]
#             #Iter只是最后一次结果，非平均的，FGD，SGD写在optimal里面
#             for i in range(10):
#                 print('times:', i)
#                 if(optimal[opt]==SGD or flag==1):
#                         start0 .append( time.time())
#                         print('origin LR:')
#                         beta_origin_FGD, iter = lr_run(X, y, optimal=optimal[opt], lamda=lamda, epoch_max=200)
#                         flag=2
#                         end0.append(time.time())
#                         loss_origin_FGD.append(l2_loss(X, y, beta_origin_FGD, lamda=lamda))
#                         iter1.append(iter)
#                 end1.append(time.time())
#                 print('sequential coreset LR:')
#                 beta_coreset_FGD,iter,timing,number =lr_coreset_run(X, y, optimal=optimal[opt], lamda=lamda, epoch_max=200)
#                 iter2.append(iter)
#                 coreset_build_time.append(timing)
#                 coreset_build_number.append(number)
#                 end2.append(time.time())
#                 print('once coreset LR:')
#                 beta_origin_SGD,iter = lr_coreset_once_run(X, y, optimal=optimal[opt], lamda=lamda, epoch_max=200)
#                 iter3.append(iter)
#                 end3 .append(time.time())
#                 print('uniform LR:')
#                 beta_coreset_SGD,iter =unisample_run(X, y, optimal=optimal[opt], lamda=lamda, epoch_max=200)
#                 iter4.append(iter)
#                 end4.append(time.time())
#
#                 loss_coreset_FGD.append(l2_loss(X,y,beta_coreset_FGD,lamda=lamda))
#                 loss_origin_SGD.append(l2_loss(X, y, beta_origin_SGD, lamda=lamda))
#                 loss_coreset_SGD.append( l2_loss(X, y, beta_coreset_SGD, lamda=lamda))
#             loss_origin_FGD, loss_coreset_FGD, loss_origin_SGD, loss_coreset_SGD,var1,var2,var3,var4 = np.mean(loss_origin_FGD), np.mean(
#                     loss_coreset_FGD), np.mean(loss_origin_SGD), np.mean(loss_coreset_SGD),np.var(loss_origin_FGD),np.var(loss_coreset_FGD),\
#                 np.var(loss_origin_SGD),np.var(loss_coreset_SGD)
#             time1,time2,time3,time4=np.mean(np.array(end0)-np.array(start0)),np.mean(np.array(end2)-np.array(end1)),np.mean(np.array(end3)-np.array(end2))\
#             , np.mean(np.array(end4) - np.array(end3))
#             iter1,iter2,iter3,iter4=np.mean(iter1),np.mean(iter2),np.mean(iter3),np.mean(iter4),
#             Coreset_time.append(np.mean(coreset_build_time))
#             Coreset_number.append(np.mean(coreset_build_number))
#             if len(Time)==0:
#                 Time=[[time1],[time2],[time3],[time4]]
#                 Loss=[[loss_origin_FGD],[loss_coreset_FGD],[loss_origin_SGD],[loss_coreset_SGD]]
#                 Loss_var=[[var1],[var2],[var3],[var4]]
#                 Iter=[[iter1],[iter2],[iter3],[iter4]]
#
#             else:
#                 Time=np.hstack((Time,[[time1],[time2],[time3],[time4]]))
#                 Loss = np.hstack((Loss, [[loss_origin_FGD],[loss_coreset_FGD],[loss_origin_SGD],[loss_coreset_SGD]]))
#                 Iter=np.hstack((Iter,[[iter1],[iter2],[iter3],[iter4]]))
#                 Loss_var = np.hstack((Loss_var,[[var1], [var2], [var3], [var4]]))
#             print('loss: loss_origin ',loss_origin_FGD,' loss_coreset ', loss_coreset_FGD ,' once coreset: ', loss_origin_SGD,' uniform: ',loss_coreset_SGD )
#             print('origin time:{},  coreset time{} ,once: {}, uniform: {}'.format(time1,time2,time3,time4))
#         Time=np.array(Time)
#         path='result/Ridge/'
#         if not os.path.exists(path):
#             os.mkdir(path)
#         scio.savemat(path+'coreset_Rd_{}.mat'.format(opt_path[opt]),{'Time':np.array(Time),'Loss_avg':np.array(Loss),'Loss_var':np.array(Loss_var),
#                                                                         'Iter':np.array(Iter),'Coreset_build_time':np.array(Coreset_time),
#                                                                         'Coreset_build_number':Coreset_number})


if __name__=="__main__":
    main()
    # main2()
    # main()