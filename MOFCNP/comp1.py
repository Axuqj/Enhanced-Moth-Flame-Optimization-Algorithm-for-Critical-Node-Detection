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
import MOFCNDP0
import MOFCNDP1

#指定默认字体
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']
matplotlib.rcParams['font.family']='sans-serif'

Name=['Random','MOFCNDP','MOFCNDP0','MOFCNDP1']

# Y = ["celegan.gml","Electronic_circuits.txt","yeast.txt","email.txt","polbooks.gml","karate.txt","lesmis.gml"]
Y = [ "email.txt","polbooks.gml","karate.txt","lesmis.gml"]
# Y1=["celegan","Electronic_circuits","yeast","email","polbooks","karate","lesmis"]
Y1=["email","polbooks","karate","lesmis"]
Z = [0.01,0.05,0.1,0.15, 0.2,0.25,0.3]
# Z = [0.01,0.05]
target="C:\\Users\\xqjwo\\Desktop\\dataset\\experiment_data1\\compresult.txt"
if os.path.getsize(target) > 0:
    with open(target, "rb") as f:
        compresult = pickle.load(f) 
else:
    compresult={}
    for nm in Name:
        compresult[nm]={y:[ [[ ],0]  for z in Z ]  for y in Y}
print(compresult)




def from_file(path):
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    G = nx.Graph()
    for line in lines:
        n = line.split()
        if not n:
            break
        G.add_edge(int(n[0]), int(n[1]))
    if not G.has_node(0): 
        G = nx.Graph()
        for line in lines:
            n = line.split()
            if not n:
                break
            G.add_edge(int(n[0])-1, int(n[1])-1)
    print( G.number_of_nodes(), G.number_of_edges())
    return G

def from_gml_file( path):
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    G = nx.Graph()
    current_edge = (-1, -1)
    for line in lines:
        words = line.split()
        if not words:
            break
        if words[0] == 'source':
            current_edge = (int(words[1]), current_edge[1])
        elif words[0] == 'target':
            G.add_edge(current_edge[0],int(words[1]))
    print( G.number_of_nodes(), G.number_of_edges())
    return G

def real(name):
    for y in Y:
        if y=="lesmis.gml" or y=="polbooks.gml" or y=="celegan.gml":            G= from_gml_file('C:\\Users\\xqjwo\\Desktop\\dataset\\'+y)
        else: G= from_file('C:\\Users\\xqjwo\\Desktop\\dataset\\'+y)
        
        for z in range(len(Z)):    #输入算法名称
            if y=="yeast.txt" and z in [0,1,2,3]:continue
            t0 = time.perf_counter()
            K=math.ceil(G.number_of_nodes()*Z[z])
            if name=='MOFCNDP' :                
                py=MOFCNDP.PyMoF(G,K)
                a   , _ ,t  =py.MFOCNP1()
                print('MOFCNDP',a) 
                t0=   time.perf_counter() - t
            if name=='MOFCNDP0' :                
                py=MOFCNDP.PyMoF(G,K)
                a   , _ ,t  =py.MFOCNP1()
                print('MOFCNDP0',a) 
                t0=   time.perf_counter() - t
            if name=='MOFCNDP1' :                
                py=MOFCNDP.PyMoF(G,K)
                a   , _ ,t  =py.MFOCNP1()
                print('MOFCNDP1',a) 
                t0=   time.perf_counter() - t             
 
            if name=='Random' :
                py=Random.PyLouvain(G )
                a=py.apply_method(K)
                print('Random',a) 
  
            time_best=round(int(time.perf_counter() - t0))
            compresult[name][y][z][0]+=[a]
            compresult[name][y][z][1]=time_best
             #result={'TBH':{'WS':(  ([   ],t),       )}}

            L = name+' '+y+str(Z[z])+' '+str(int(a))+' '+str(time_best)
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
    
    f = open("C:\\Users\\xqjwo\\Desktop\\dataset\\experiment_data1\\comp_connectivity.txt", "a+")
    f.write(L+'\n')
    f.write(L1+'\n')
    f.close()

    M0=[]      
    inum=0
    for i in compresult: #遍历算法 i 代表算法，compresult[i]代表算法对应的结果，compresult[i][j]代表一个算法的结果
        if i =='Random':continue #不输入random
        M0+=[[]] 
        inum+=1
        for j in compresult[i]: # 遍历一算法的每个网络
            for x in compresult[i][j]:
                M0[inum-1]+= [   max(x[0])     ]
    K0=np.transpose(M0)

    M1=[]      
    inum=0
    for i in compresult: #遍历算法 i 代表算法，compresult[i]代表算法对应的结果，compresult[i][j]代表一个算法的结果
        if i =='Random':continue #不输入random
        M1+=[[]] 
        inum+=1
        for j in compresult[i]: # 遍历一算法的每个网络
            for x in compresult[i][j]:
                M1[inum-1]+= [   np.mean(x[0])     ]
    K=np.transpose(M1)

    M2=[]       
    inum=0
    for i in compresult: #遍历算法 i 代表算法，compresult[i]代表算法对应的结果，compresult[i][j]代表一个算法的结果
        if i =='Random':continue #不输入random
        M2+=[[]] 
        inum+=1
        for j in compresult[i]: # 遍历一算法的每个网络
            for x in compresult[i][j]:
                M2[inum-1]+= [x[1]]
    K2=np.transpose(M2)

    f = open("C:\\Users\\xqjwo\\Desktop\\dataset\\experiment_data1\\comp_connectivity.txt", "a+")    #
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
    # print(compresult)
    for i in compresult: #遍历算法 compresult[i]代表一种算法random，compresult[i][j]代表一个算法的结果
        if i !='Random':
            M+=[[]]
            inum+=1
        for j in compresult[i]: # 遍历一算法的每个网络 compresult[i]=random 表示random有多少个网络
            if i =='Random':
                r+=[min(compresult[i][j][0][0]) , min(compresult[i][j][1][0])]
                # r+=[compresult[i][j][2], compresult[i][j][6]]  #存储random的每个值，生成[1,2,3,4,5,6,] 取0.1 0.3
                continue
            M[inum-1]+= [min(compresult[i][j][0][0]) , min(compresult[i][j][1][0])]
            # M[inum-1]+= [compresult[i][j][2], compresult[i][j][6]] #M[m]表示第m个算法的所有值 [ [1,2,3,4,5], [1,2,3,4,5]]
        N=[i for i in M if len(i)!=0]
        if i =='Random':continue  #random不参与百分比计算
        for p in range(len(M[inum-1])):
            num=round(100*(r[p]-M[inum-1][p])/r[p] ,2 )
            N[inum-1][p]=num
            if num<0:N[inum-1][p]=  0
    
    K= [list(d) for d in np.transpose(N)]
    index1=[[i+'1',i+'2'] for i in Y1]
    Name1=[i for i in Name if i !='Random']
    index1=[i  for j in index1 for i in j  ]
    # print(K)
    matplotlib.rcParams['axes.unicode_minus'] = False
    df = pd.DataFrame(K,
                    index=index1,
                    columns=pd.Index(Name1),
                    )

    df.plot(kind='barh') #,figsize=(5,8)

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.legend(loc="best")
    plt.xlabel("The relative error")
    plt.ylabel("Networks")
    plt.savefig('C:\\Users\\xqjwo\\Desktop\\dataset\\experiment_data1\\comp_random.pdf',
                bbox_inches='tight',
                dpi=400,
                )
    plt.show()

def pp():   #性能概况
    M1=[]       
    m=0
    for i in compresult: #遍历算法 i 代表算法，compresult[i]代表算法对应的结果，compresult[i][j]代表一个算法的结果
        if i =='Random':continue #不输入random
        M1+=[[]] 
        m+=1 
        for j in compresult[i]: # 遍历一算法的每个网络
            M1[m-1]+= [  min(x[0])   for x in compresult[i][j] ]
    
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
    
    plt.legend()#绘制图例
    plt.xlabel('n')
    plt.ylabel('P(n)')

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    plt.xlim(0,n[-1])
    plt.ylim(0,1.0)

    plt.savefig('C:\\Users\\xqjwo\\Desktop\\real_pp.pdf',#'C:\\Users\\xqjwo\\Desktop\\dataset\\experiment_data1\\real_pp.pdf'
                bbox_inches='tight',
                dpi=300,
                )
    plt.show()


def experiment():
    for name in Name:
        real(name)
    with open("C:\\Users\\xqjwo\\Desktop\\dataset\\experiment_data1\\compresult.txt", "wb") as f:
        pickle.dump(compresult, f)    
    print(compresult)


    # with open("C:\\Users\\xqjwo\\Desktop\\dataset\\experiment_data1\\compresult.txt", "rb") as f:
    #     compresult = pickle.load(f)
    # compresult['OCH']={Y1[0]:(),Y1[1]:()}
    # print(compresult['OCH'])
    # real('OCH')
    # with open("C:\\Users\\xqjwo\\Desktop\\dataset\\experiment_data1\\compresult.txt", "a+") as f:
    #     pickle.dump(compresult, f)    
    # print(compresult)   

    # real('TBH')
    # with open("C:\\Users\\xqjwo\\Desktop\\dataset\\experiment_data1\\compresult.txt", "wb") as f:
    #     pickle.dump(compresult, f)    
    # print(compresult)

def draw():
    connectivity()
    # Rd()
    # pp()
    # lines()
    # line()


if __name__ == "__main__":
    experiment()#运行，写入文件
    # draw()#读取文件，画图



