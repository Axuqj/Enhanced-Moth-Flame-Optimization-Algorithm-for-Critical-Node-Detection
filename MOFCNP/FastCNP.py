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

    def apply_method(self,K,time_max,pwp):
        S=[]
        node_num=self.G.number_of_nodes()
         
        #初始集 
        while len(S) < K:
            rdn=random.choice(list(self.G.nodes-S))
            if rdn not in S:
                S.append(rdn)
        # if len(S) == K:print('ok')

        S = {i:1 for i in S}
        S1={i:1 for i in S}
        t0=time.time() #开始计时
        pw = math.ceil(K*pwp) #perturbation strength

        while time.time()-t0 < time_max:
            #fast neighborhood search
            for i in range(30):
                new_node=random.choice(list(self.G.nodes-S.keys()))
                S[new_node]=1 

                L = sorted(S.keys(),key=lambda x: self.f(S.keys()-{x} )) 
                S.pop(L[0])

                if self.f(S)>self.f(S1): #保证结果有进步，否则回撤操作
                    S=S1
                else:
                    S1=S
                    # fmin=fnew
 
            #destructive-constructive
            pcount=0
            while pcount<pw:
                # print(len(S),pw)
                new_node=random.choice(list(S.keys()))
                S.pop(new_node)
                pcount+=1

            pcount=0
            while pcount<pw:
                new_node=random.choice(list(self.G.nodes-S.keys()))
                S[new_node]=1    
                pcount+=1

        result=self.f(S1 )
        print(S1.keys())
        print(K,len(S1 )) 
        return result 

if __name__ == "__main__":
    Y=['WattsStrogatz_n250.txt','BarabasiAlbert_n500m1.txt','ErdosRenyi_n250.txt']
    Z=[70,50,50]
    i=         2
    time1=200
    pwp=0.03
    pyl=PyLouvain.from_adjlist("C:\\Users\\xqjwo\\Desktop\\dataset\\新建文件夹\\ACO_CNP-master\\instancs\\"+Y[i])
    print(Y[i],pyl.apply_method(Z[i],time1,pwp))

