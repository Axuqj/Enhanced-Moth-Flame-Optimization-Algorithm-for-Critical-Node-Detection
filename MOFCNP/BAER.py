import random
import TBH
import numpy as np
import sys
import GCNP
import math
import os
import Random
import matplotlib.pyplot as plt
import HDCN
import greedy3d
import greedy4d
import MOFCNDP
from scipy import interpolate
import time
import pickle
import pandas as pd
import seaborn as sns
import matplotlib
import networkx as nx
import xlrd
 
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']
matplotlib.rcParams['font.family']='sans-serif'
#Name=['Random','OCH','TBH','GCNP','BCB','HCH','KSBG','LRBG','SG']
Name=['Random','MOFCNDP','GCNP','greedy3d','greedy4d','HDCN','KSBG','LRBG','BCB']

target="C:\\Users\\xqjwo\\Desktop\\dataset\\experiment_data1\\BAresult1.txt"
if os.path.getsize(target) > 0:  #每次启动 都会注入result
    with open(target, "rb") as f:
        Rresult = pickle.load(f) 
else:
    Rresult={name:{'WS':[[[],0],[[],0],[[],0]],'ER':[[[],0],[[],0],[[],0]],'BA':[[[],0],[[],0],[[],0]]}  for name in Name}#存储结果
print(Rresult)
Y={ 'ER':['ErdosRenyi_n250.txt','ErdosRenyi_n500.txt','ErdosRenyi_n1000.txt'], 
'BA':['BarabasiAlbert_n500m1.txt','BarabasiAlbert_n1000m1.txt','BarabasiAlbert_n2500m1.txt'] ,
 'WS':['WattsStrogatz_n250.txt','WattsStrogatz_n500.txt','WattsStrogatz_n1000.txt']   }
Z={ 'ER':[50,80,140],'BA':[50,75,100],'WS':[70,125,200]  }


def from_adjlist(path): #接收adjlist  
    G=nx.read_adjlist(path, nodetype=int)
    return G

def BA_ER(name):
    for y1 in Y:
        for y2 in range(len(Y[y1])):  
            t0 = time.perf_counter()
            G=from_adjlist("C:\\Users\\xqjwo\\Desktop\\dataset\\新建文件夹\\ACO_CNP-master\\instancs\\"+Y[y1][y2])
            if name=='MOFCNDP' :                
                py=MOFCNDP.PyMoF(G,Z[y1][y2])
                a   , _ ,t  =py.MFOCNP1()
                print('MOFCNDP',a) 
                t0=   time.perf_counter() - t
            if name=='greedy3d' :
                py=greedy3d.PyLouvain(G)
                a =py.apply_method(Z[y1][y2])
                print('greedy3d',a)
            if name=='greedy4d' :
                py=greedy4d.PyLouvain(G )
                a =py.apply_method(Z[y1][y2])
                print('greedy4d',a)
            if name=='GCNP' :
                py=GCNP.PyLouvain(G )
                a  =py.apply_method(Z[y1][y2])
                print('GCNP',a)
            if name=='Random' :
                py=Random.PyLouvain(G )
                a=py.apply_method(Z[y1][y2])
                print('Random',a) 
            if name=='HDCN' :
                py=HDCN.PyLouvain(G )
                a =py.apply_method(Z[y1][y2])
                print('HDCN',a) 
            time_best=round(int(time.perf_counter() - t0))
            Rresult[name][y1][y2][0]+=[a]
            Rresult[name][y1][y2][1]=time_best
             #result={'TBH':{'WS':(  ([   ],t),       )}}

            L = name+' '+y1+str(Z[y1][y2])+' '+str(int(a))+' '+str(time_best)
            f = open("C:\\Users\\xqjwo\\Desktop\\dataset\\experiment_data1\\test.txt", "a+")
            f.write(L+'\n')
            f.close()

def connectivity():        #连通节点对 表示算法name在网络y的第i个参数的结果是c：result={'CBG':{'network1': (1,2,3), '':()}, ''   }

    L='\\multirow{'+'2'+'}{'+'*}{ Network }'  
    for n in range(len(Name)):  #  \multicolumn{2}{c}{CBG}
        if Name[n] =='Random':continue
        L+=   '\\multicolumn{'+'3'+'}{'+'c}{' +Name[n]+'}'
        if n!=len(Name)-1 : L+=' & ' 
        else: L+=' \\\\'
    L1=''  
    for n in range(len(Name)):  #  \multicolumn{2}{c}{CBG}
        if Name[n] =='Random':continue
        L1+=  '& \multicolumn{ 1}{l}{fbest} & \multicolumn{ 1}{l}{fmin} & \multicolumn{1}{l}{tbest}'
        if n!=len(Name)-1 : L1+=' & ' 
        else: L1+=' \\\\'
    
    f = open("C:\\Users\\xqjwo\\Desktop\\dataset\\experiment_data1\\BA_connectivity.txt", "a+")
    f.write(L+'\n')
    f.write(L1+'\n')
    f.close()

    M0=[]      
    inum=0
    for i in Rresult: #遍历算法 i 代表算法，Rresult[i]代表算法对应的结果，Rresult[i][j]代表一个算法的结果
        if i =='Random':continue #不输入random
        M0+=[[]] 
        inum+=1
        for j in Rresult[i]: # 遍历一算法的每个网络
            for x in Rresult[i][j]:
                M0[inum-1]+= [   min(x[0])     ]
    K0=np.transpose(M0)

    M1=[]      
    inum=0
    for i in Rresult: #遍历算法 i 代表算法，Rresult[i]代表算法对应的结果，Rresult[i][j]代表一个算法的结果
        if i =='Random':continue #不输入random
        M1+=[[]] 
        inum+=1
        for j in Rresult[i]: # 遍历一算法的每个网络
            for x in Rresult[i][j]:
                M1[inum-1]+= [   np.mean(x[0])     ]
    K=np.transpose(M1)

    M2=[]       
    inum=0
    for i in Rresult: #遍历算法 i 代表算法，Rresult[i]代表算法对应的结果，Rresult[i][j]代表一个算法的结果
        if i =='Random':continue #不输入random
        M2+=[[]] 
        inum+=1
        for j in Rresult[i]: # 遍历一算法的每个网络
            for x in Rresult[i][j]:
                M2[inum-1]+= [x[1]]
    K2=np.transpose(M2)

    f = open("C:\\Users\\xqjwo\\Desktop\\dataset\\experiment_data1\\BA_connectivity.txt", "a+")    #
    for i in range(len(K)): #i换成K[i]
        L=''
        mini=min(K[i])
        minj=min(K0[i])
        for j in range(len(K[i])):
            if K0[i][j]==minj: L+= '\\textbf{' + str(int(K0[i][j]))+'}'+' & '
            else:L+=str(int(K0[i][j]))+' & '
            if K[i][j]==mini: L+= '\\textbf{' + str(int(K[i][j]))+'}'+' & '+ str(int(K2[i][j]))
            else:L+=str(int(K[i][j]))+' & '+ str(int(K2[i][j]))
            if j!=len(K[i])-1 : L+=' & ' 
            else: L+=' \\\\'
        f.write(L+'\n')
    f.close()

def Rd():
    M=[]
    N=[]
    r=[]   
    inum=0  #m跟i对标，n跟j对标
    for i in Rresult: #遍历算法 Rresult[i]代表一种算法random，Rresult[i][j]代表一个算法的结果
        if i !='Random':
            M+=[[]]
            inum+=1
        for j in Rresult[i]: # 遍历一算法的每个网络 Rresult[i]=random 表示random有多少个网络
            if i =='Random' or j=='WS':
                r+=[  min(Rresult[i][j][0][0]) , min(Rresult[i][j][1][0]),min(Rresult[i][j][2][0])]  #存储random的每个值，生成[1,2,3,4,5,6,]
                continue
            M[inum-1]+= [ min(Rresult[i][j][0][0]) , min(Rresult[i][j][1][0]),min(Rresult[i][j][2][0])]
            # M[inum-1]+= [Rresult[i][j][2], Rresult[i][j][6]] #M[m]表示第m个算法的所有值 [ [1,2,3,4,5], [1,2,3,4,5]]
        N=[i for i in M if len(i)!=0]
        if i =='Random'or j=='WS':continue  #random不参与百分比计算
        for p in range(len(M[inum-1])):
            num=round(100*(r[p]-M[inum-1][p])/r[p] ,3 )
            N[inum-1][p]=num
            if num<0:N[inum-1][p]=  0
            # if num<100: num=num*100
    
    K= [list(d) for d in np.transpose(N)]
    index1=[[],[]]
    for i in Y:
        if i =='BA':
            for y in Y[i]:
                index1[0]+= [i+y.split()[0]]
        if i =='ER':
            for y in Y[i]:
                index1[1]+= [i+y.split()[0]]
    print(Rresult)
    
    Name1=[i for i in Name if i !='Random' ]
    # print(Name1)
    index1=[i  for j in index1 for i in j  ]
    # print(N)
    # sys.exit()
    matplotlib.rcParams['axes.unicode_minus'] = False
    df = pd.DataFrame(K,
                    index=index1,
                    columns=pd.Index(Name1),
                    )

    df.plot(kind='barh') #,figsize=(5,8)

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.legend(loc="upper left")
    plt.xlabel("The relative error")
    plt.ylabel("Networks")
    plt.savefig('C:\\Users\\xqjwo\\Desktop\\BA_random.pdf',#C:\\Users\\xqjwo\\Desktop\\dataset\\experiment_data1\\BA_random.pdf
                bbox_inches='tight',
                dpi=400,
                )
    plt.show()

def pp():   #性能概况
    M1=[]
    m=0
    for i in Rresult: #遍历算法 i 代表算法，Rresult[i]代表算法对应的结果，Rresult[i][j]代表一个算法的结果
        if i =='Random':continue #不输入random
        M1+=[[]] 
        m+=1 
        for j in Rresult[i]: # 遍历一算法的每个网络
            M1[m-1]+=  [  min(x[0])    for x  in Rresult[i][j] ] 
    
    # print(M1)
    best=[min(m) for m in np.transpose(M1)]   #最优值集合
    # print(best)
    
    inum=len(best) #案例个数 
    M2=M1.copy()
    for i in range(len(M1)):
        M2[i]=[round(math.log(M1[i][j]/best[j],2) , 2)  for j in range(len(M1[i])) ]  #转化成log2(T/best)，后面就很好算。。
    nmax=np.max(M2)

    M3=[[0 for i in range(100)] for j in range(len(M2))]  #M3[i]就是每个算法的p（n）
    for i in range(len(M2)): #m是算法如cbg的序号
        for j in range(0,100):
            M3[i][j]= round(np.sum(np.array(M2[i])<=j*nmax/100)/len(M2[i]),2) #大于当下n的多少，也就是p（n)!!!
    
    # lines=['-','--' , '-.']
    markers=['.','o' ,'+'  ,'^', '*','x' ,''  ]
    colors=['b','g','r','c','m','y','k','r']

    n=[i*nmax/100 for i in range(100) ]
    
    for d in range(len(Name)-1):
        plt.plot(n,#x轴方向变量
                interpolate.interp1d(n,M3[d],kind='cubic')(n),#y轴方向变量
                # linestyle=random.choice(lines),#线型
                color=colors[d],#线和marker的颜色random.choice(colors)
                # marker=markers[d],#marker形状random.choice(markers)
                label=Name[d+1],#图例
                )
    
    plt.legend(loc="best")#绘制图例
    plt.xlabel('n')
    plt.ylabel('P(n)')

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    plt.xlim(0,n[-1])

    plt.savefig('C:\\Users\\xqjwo\\Desktop\\BA_pp.pdf',#'C:\\Users\\xqjwo\\Desktop\\dataset\\experiment_data1\\BA_pp.pdf'
                bbox_inches='tight',
                dpi=300,
                )
    plt.show()

def read1():
    Name=['MOFCNDP','GCNP','greedy3d','greedy4d','HDCN','KSBG','LRBG','BCB']	
    network=['ER','BA','WS']	

    Rresult={name:{'ER':[[[],0],[[],0],[[],0],[[],0]],'BA':[[[],0],[[],0],[[],0],[[],0]],'WS':[[[],0],[[],0],[[],0],[[],0]]}  for name in Name}

    data = xlrd.open_workbook(r'C:\\Users\\xqjwo\\Desktop\\new artical\\1.xls')
    table = data.sheets()[0]

    line=range(2,10)
    row=range(110,122)

    for i in range(len(line)): #列，也是算法序号
        for j in range(len(row)):
            ntw= network[int(j/4)]
            no=j%4
            Rresult[Name[i]][ntw][no][0] = [int(table.cell_value(row[j],line[i]))]

    print(Rresult) 
    with open("C:\\Users\\xqjwo\\Desktop\\dataset\\experiment_data1\\BAresult1.txt", "wb") as f:
        pickle.dump(Rresult, f)  


def experiment():
    for name in Name:
        BA_ER(name)
    # BA_ER('GCNP')
    # BA_ER('TBH')

def draw():
    connectivity()
    # Rd()
    # pp()

if __name__ == "__main__":
    # with open("C:\\Users\\xqjwo\\Desktop\\dataset\\experiment_data1\\BAresult1.txt", "rb") as f:
        # Rresult = pickle.load(f)
    # print(Rresult)    
    # experiment()#运 行，写入文件
    read1()
    draw()#读取文件，画图
    
