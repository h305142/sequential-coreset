import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import os
from matplotlib.pyplot import MultipleLocator
filenames = ['gmm_d.mat',
            'gmm_k.mat',
             'gmm_size.mat',
             ]

xlabelSize = 32
ylabelSize = 32
xticksSize = 32
yticksSize = 32
titleSize = 28
legendSize = 20

datapath = os.getcwd() +  '/../result/'

# Varying the dimension d
dname = filenames[0]
record1 = scio.loadmat(datapath+dname)

# time
Timer_avg_table = record1['Timer_avg_table']
Timer_std_table = record1['Timer_std_table']

# y_avg = Timer_avg_table/Timer_avg_table[:,0][:,None]
# y_std = Timer_std_table/(Timer_avg_table[:,0][:,None]*Timer_avg_table[:,0][:,None])
y_avg = Timer_avg_table
y_std = Timer_std_table

fig = plt.figure(figsize=(8,6))
plt.title('GMM $(k = 12, r = 2\%)$',fontsize=titleSize)
plt.xlabel('Dimension $d$',fontsize=xlabelSize)
plt.ylabel('Runtime (s)',fontsize=ylabelSize)
plt.yticks(fontsize=yticksSize)
cell = [r'4',r'8',r'12',r'16',r'20',r'24']
x = np.array([4,8,12,16,20,24])
plt.xticks([4,8,12,16,20,24],cell,fontsize=xticksSize)
# plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0],\
#   ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
plt.yticks([0, 10, 20, 30, 40, 50],\
  ['0', '10', '20', '30', '40', '50'])
# plt.ylim(0,0.3)
plt.ylim(0,60)

ln0 = plt.errorbar(x,y_avg[:,0],yerr=y_std[:,0],color='g',linewidth=2.0,linestyle='-',marker='+',label="Original")
ln1 = plt.errorbar(x,y_avg[:,1],yerr=y_std[:,1],color='y',linewidth=2.0,linestyle='-',marker='d',label="UniSamp")
ln2 = plt.errorbar(x,y_avg[:,2],yerr=y_std[:,2],color='m',linewidth=2.0,linestyle='-',marker='s',label="ImpSamp")
ln3 = plt.errorbar(x,y_avg[:,3],yerr=y_std[:,3],color='k',linewidth=2.0,linestyle='--',marker='o',label="SeqCore")

plt.legend(bbox_to_anchor=(0.01, 0.99), loc=2, borderaxespad=0, fontsize=legendSize)
plt.tight_layout()
plt.savefig('../figures/gmm_d_time'+'.eps',format="eps")
plt.savefig('../figures/gmm_d_time.png')
plt.show()

# purity
Purity_avg_table = record1['Purity_avg_table']
Purity_std_table = record1['Purity_std_table']

y_avg = Purity_avg_table
y_std = Purity_std_table

fig = plt.figure(figsize=(8,6))
plt.title('GMM $(k = 12, r = 2\%)$',fontsize=titleSize)
plt.xlabel('Dimension $d$',fontsize=xlabelSize)
plt.ylabel('Purity',fontsize=ylabelSize)
plt.yticks(fontsize=yticksSize)
cell = [r'4',r'8',r'12',r'16',r'20',r'24']
x = np.array([4,8,12,16,20,24])
plt.xticks([4,8,12,16,20,24],cell,fontsize=xticksSize)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0],\
  ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
plt.ylim(0.4,1.2)

ln0 = plt.errorbar(x,y_avg[:,0],yerr=y_std[:,0],color='g',linewidth=2.0,linestyle='-',marker='+',label="Original")
ln1 = plt.errorbar(x,y_avg[:,1],yerr=y_std[:,1],color='y',linewidth=2.0,linestyle='-',marker='d',label="UniSamp")
ln2 = plt.errorbar(x,y_avg[:,2],yerr=y_std[:,2],color='m',linewidth=2.0,linestyle='-',marker='s',label="ImpSamp")
ln3 = plt.errorbar(x,y_avg[:,3],yerr=y_std[:,3],color='k',linewidth=2.0,linestyle='--',marker='o',label="SeqCore")

plt.legend(bbox_to_anchor=(0.01, 0.99), loc=2, borderaxespad=0, fontsize=legendSize)
plt.tight_layout()
plt.savefig('../figures/gmm_d_purity'+'.eps',format="eps")
plt.savefig('../figures/gmm_d_purity.png')
plt.show()


# Varying the components k
dname = filenames[1]
record1 = scio.loadmat(datapath+dname)

# time
Timer_avg_table = record1['Timer_avg_table']
Timer_std_table = record1['Timer_std_table']

# y_avg = Timer_avg_table/Timer_avg_table[:,0][:,None]
# y_std = Timer_std_table/(Timer_avg_table[:,0][:,None]*Timer_avg_table[:,0][:,None])
y_avg = Timer_avg_table
y_std = Timer_std_table

fig = plt.figure(figsize=(8,6))
plt.title('GMM $(d = 12, r = 2\%)$',fontsize=titleSize)
plt.xlabel('Gaussian Components $k$',fontsize=xlabelSize)
plt.ylabel('Runtime (s)',fontsize=ylabelSize)
plt.yticks(fontsize=yticksSize)
cell = [r'4',r'8',r'12',r'16',r'20',r'24']
x = np.array([4,8,12,16,20,24])
plt.xticks([4,8,12,16,20,24],cell,fontsize=xticksSize)
# plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0],\
#   ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
plt.yticks([0, 20, 40, 60, 80],\
  ['0', '20', '40', '60', '80'])
# plt.ylim(0,0.3)
plt.ylim(0,100)

ln0 = plt.errorbar(x,y_avg[:,0],yerr=y_std[:,0],color='g',linewidth=2.0,linestyle='-',marker='+',label="Original")
ln1 = plt.errorbar(x,y_avg[:,1],yerr=y_std[:,1],color='y',linewidth=2.0,linestyle='-',marker='d',label="UniSamp")
ln2 = plt.errorbar(x,y_avg[:,2],yerr=y_std[:,2],color='m',linewidth=2.0,linestyle='-',marker='s',label="ImpSamp")
ln3 = plt.errorbar(x,y_avg[:,3],yerr=y_std[:,3],color='k',linewidth=2.0,linestyle='--',marker='o',label="SeqCore")

plt.legend(bbox_to_anchor=(0.01, 0.99), loc=2, borderaxespad=0, fontsize=legendSize)
plt.tight_layout()
plt.savefig('../figures/gmm_k_time'+'.eps',format="eps")
plt.savefig('../figures/gmm_k_time.png')
plt.show()

# purity
Purity_avg_table = record1['Purity_avg_table']
Purity_std_table = record1['Purity_std_table']

y_avg = Purity_avg_table
y_std = Purity_std_table

fig = plt.figure(figsize=(8,6))
plt.title('GMM $(d = 12, r = 2\%)$',fontsize=titleSize)
plt.xlabel('Gaussian Components $k$',fontsize=xlabelSize)
plt.ylabel('Purity',fontsize=ylabelSize)
plt.yticks(fontsize=yticksSize)
cell = [r'4',r'8',r'12',r'16',r'20',r'24']
x = np.array([4,8,12,16,20,24])
plt.xticks([4,8,12,16,20,24],cell,fontsize=xticksSize)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],\
  ['0', '0.2', '0.4', '0.6', '0.8', '1.0', '1.2'])
plt.ylim(0.1,1.4)

ln0 = plt.errorbar(x,y_avg[:,0],yerr=y_std[:,0],color='g',linewidth=2.0,linestyle='-',marker='+',label="Original")
ln1 = plt.errorbar(x,y_avg[:,1],yerr=y_std[:,1],color='y',linewidth=2.0,linestyle='-',marker='d',label="UniSamp")
ln2 = plt.errorbar(x,y_avg[:,2],yerr=y_std[:,2],color='m',linewidth=2.0,linestyle='-',marker='s',label="ImpSamp")
ln3 = plt.errorbar(x,y_avg[:,3],yerr=y_std[:,3],color='k',linewidth=2.0,linestyle='--',marker='o',label="SeqCore")

plt.legend(bbox_to_anchor=(0.99, 0.99), loc=1, borderaxespad=0, fontsize=legendSize)
plt.tight_layout()
plt.savefig('../figures/gmm_k_purity'+'.eps',format="eps")
plt.savefig('../figures/gmm_k_purity.png')
plt.show()


# Varying the coreset size
dname = filenames[2]
record1 = scio.loadmat(datapath+dname)

# time
Timer_avg_table = record1['Timer_avg_table']
Timer_std_table = record1['Timer_std_table']

# y_avg = Timer_avg_table/Timer_avg_table[:,0][:,None]
# y_std = Timer_std_table/(Timer_avg_table[:,0][:,None]*Timer_avg_table[:,0][:,None])
y_avg = Timer_avg_table
y_std = Timer_std_table

fig = plt.figure(figsize=(8,6))
plt.title('GMM $(d = 12, k = 12)$',fontsize=titleSize)
plt.xlabel('Sampling Ratio $r$ $(\%)$',fontsize=xlabelSize)
plt.ylabel('Runtime (s)',fontsize=ylabelSize)
plt.yticks(fontsize=yticksSize)
cell = [r'2',r'4',r'6',r'8',r'10']
x = np.array([1,2,3,4,5,6,7,8,9,10])
plt.xticks([2,4,6,8,10],cell,fontsize=xticksSize)
# plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0],\
#   ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
plt.yticks([0, 1, 2, 3, 4, 5],\
  ['0', '1', '2', '3', '4', '5'])
# plt.ylim(0,0.3)
plt.ylim(0,6)

# ln0 = plt.errorbar(x,y_avg[:,0],yerr=y_std[:,0],color='g',linewidth=2.0,linestyle='-',marker='+',label="Original")
ln1 = plt.errorbar(x,y_avg[:,1],yerr=y_std[:,1],color='y',linewidth=2.0,linestyle='-',marker='d',label="UniSamp")
ln2 = plt.errorbar(x,y_avg[:,2],yerr=y_std[:,2],color='m',linewidth=2.0,linestyle='-',marker='s',label="ImpSamp")
ln3 = plt.errorbar(x,y_avg[:,3],yerr=y_std[:,3],color='k',linewidth=2.0,linestyle='--',marker='o',label="SeqCore")

plt.legend(bbox_to_anchor=(0.01, 0.99), loc=2, borderaxespad=0, fontsize=legendSize)
plt.tight_layout()
plt.savefig('../figures/gmm_size_time'+'.eps',format="eps")
plt.savefig('../figures/gmm_size_time.png')
plt.show()

# purity
Purity_avg_table = record1['Purity_avg_table']
Purity_std_table = record1['Purity_std_table']

y_avg = Purity_avg_table
y_std = Purity_std_table

fig = plt.figure(figsize=(8,6))
plt.title('GMM $(d = 12, k = 12)$',fontsize=titleSize)
plt.xlabel('Sampling Ratio $r$ $(\%)$',fontsize=xlabelSize)
plt.ylabel('Purity',fontsize=ylabelSize)
plt.yticks(fontsize=yticksSize)
cell = [r'2',r'4',r'6',r'8',r'10']
x = np.array([1,2,3,4,5,6,7,8,9,10])
plt.xticks([2,4,6,8,10],cell,fontsize=xticksSize)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],\
  ['0', '0.2', '0.4', '0.6', '0.8', '1.0', '1.2'])
plt.ylim(0.1,1.2)

# ln0 = plt.errorbar(x,y_avg[:,0],yerr=y_std[:,0],color='g',linewidth=2.0,linestyle='-',marker='+',label="Original")
ln1 = plt.errorbar(x,y_avg[:,1],yerr=y_std[:,1],color='y',linewidth=2.0,linestyle='-',marker='d',label="UniSamp")
ln2 = plt.errorbar(x,y_avg[:,2],yerr=y_std[:,2],color='m',linewidth=2.0,linestyle='-',marker='s',label="ImpSamp")
ln3 = plt.errorbar(x,y_avg[:,3],yerr=y_std[:,3],color='k',linewidth=2.0,linestyle='--',marker='o',label="SeqCore")

plt.legend(bbox_to_anchor=(0.01, 0.99), loc=2, borderaxespad=0, fontsize=legendSize)
plt.tight_layout()
plt.savefig('../figures/gmm_size_purity'+'.eps',format="eps")
plt.savefig('../figures/gmm_size_purity.png')
plt.show()