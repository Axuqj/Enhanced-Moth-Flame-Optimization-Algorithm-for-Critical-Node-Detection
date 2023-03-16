#!/usr/bin/env python3
import copy
from os import system
import networkx as nx
import numpy as np
import math
import time
import operator
import sys
import random 

class PyLouvain:
    @classmethod
    def from_adjlist(cls, path): #接收adjlist  
        G=nx.read_adjlist(path, nodetype=int)
        return cls(G)

    @classmethod
    def from_file(cls, path): #接收txt 
        f = open(path, 'r')
        lines = f.readlines()
        f.close()
        G = nx.Graph()
        k=-2 #记录当前行数
        for line in lines:
            k+=1
            if k==-1: continue #忽视第一行，第二行lines[1]才取用，此时k=0
            n = line.split()
            if not n:
                break
            for i in range(len(n)):
                if i==0:
                    continue #忽视第一个数
                G.add_edge(k, int(n[i])) #录入边
        return cls(G)

    def __init__(self, G):
        self.G=G
        self.G1=self.G.copy()
        print('nodes:',self.G.number_of_nodes(), 'edges:',self.G.number_of_edges())
    
    def f(self,s): #计算graph中除去s后，的连通对数.所有s都排除
        G1=self.G.subgraph(self.G.nodes-s)
        cp = {}#cp:connection_partition
        visited = []
        stack = []
        id = 0
        cp[0]=[]
        num=0

        def dfs(v):  #搜索连通分量
            cp[id].append(v)
            visited.append(v) 
            stack.append(v)
            while stack:
                node = stack.pop()
                for neighbour in G1.neighbors(node): 
                    if neighbour not in visited: 
                        visited.append(neighbour) 
                        cp[id].append(neighbour) 
                        stack.append(neighbour)
            return len(cp[id])

        for v in G1.nodes:
            if v not in visited:                 
                cp[id]=[]
                n = dfs(v)
                num += n*(n-1)/2
                id += 1
        return num

    def apply_method(self,K):
        S=[]
        # K = math.ceil(self.G.number_of_nodes()*z)
        node_num=self.G.number_of_nodes()
        
        #初始集 #1.去掉度为0或1的点 2.随机删点直到没边
        # S0=[]
        # S0=[i for i in self.G.nodes if self.G.degree(i)==1 or self.G.degree(i)==0 ]
        while 1: 
            self.G1=self.G.subgraph(self.G.nodes-S)
            if self.G1.number_of_edges()==0: #直到没有边了
                break            
            # L = sorted(self.G1.nodes,key=lambda x:self.G1.degree(x),reverse=True)
            while 1:
                i=random.randint(0, node_num-1)                
                if i not in S:
                    S.append(i)
                    break
        S = {i:1 for i in S}
        S1={i:1 for i in S}
        
        k = math.ceil(K/2)
        k_1=math.ceil(K-k)
        k1=math.ceil(K+k) 

        S={i:1 for i in S1}
        S1={i:1 for i in S1}

        N=0
        Nc=60 #迭代次数
        fmin= (node_num-1)*node_num/2
        while N < Nc: 
            while len(S1) < k1 : # 增
                y = lambda x: self.f(S1.keys() | {x} )
                L = sorted(self.G.nodes-S1.keys(),key= y)
                S1[L[0]]=1

            while len(S1) >K :  #删
                y = lambda x: self.f(S1.keys()-{x} )
                L = sorted(S1.keys(),key= y) 
                S1.pop(L[0])
            
            N+=1
            fnew=self.f(S1)
            if fnew>fmin: #保证结果有进步，否则回撤操作
                S1=S
            else:
                S=S1
                fmin=fnew

            while len(S1) > k_1 : #删
                y = lambda x: self.f(S1.keys()-{x} )
                L = sorted(S1.keys(),key= y)
                S1.pop(L[0])

            while len(S1) < K :  #增
                y = lambda x: self.f(S1.keys() | {x} )
                L = sorted(self.G.nodes-S1.keys(),key= y)
                S1[L[0]]=1
                            
            N += 1 
            fnew=self.f(S1)
            if fnew>fmin: #保证结果有进步，否则回撤操作
                S1=S
            else:
                S=S1
                fmin=fnew
     
        result=self.f(S )
        print(K,len(S )) 
        return result

name='HCH'
if __name__ == "__main__":
    Y=['WattsStrogatz_n250.txt','BarabasiAlbert_n500m1.txt','ErdosRenyi_n250.txt']
    Z=[70,50,50]
    i=0
    pyl=PyLouvain.from_adjlist("C:\\Users\\xqjwo\\Desktop\\dataset\\新建文件夹\\ACO_CNP-master\\instancs\\"+Y[i])
    print(pyl.apply_method(Z[i]))
    # for i in range(1):
    #     pyl = PyLouvain.from_file("C:\\Users\\xqjwo\\Desktop\\dataset\\标准数据集\\"+Y[i])
    #     print(pyl.apply_method(Z[i]))







