#!/usr/bin/env python3
import copy
import networkx as nx
import numpy as np
import sys
import time
import math

class PyLouvain:
    @classmethod
    def from_ERgraph(cls, n,m): # n为点数，m为边
        p = 2*m/(n*(n-1))
        G = nx.random_graphs.erdos_renyi_graph(n,p)
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
    @classmethod
    def from_WSgraph(cls, a,b,p): # a为点数，b为邻居数，p为重连概率
        G = nx.random_graphs.watts_strogatz_graph(a,b,p)
        return cls(G )

    @classmethod
    def from_BAgraph(cls, n,m): # n为点数，m为每次加入数
        G = nx.random_graphs.barabasi_albert_graph(n,m)
        return cls(G )

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

    def apply_method(self,k):
        S = []

        #初始集
        while 1: 
            self.G1=self.G.subgraph(self.G.nodes-S)
            if self.G1.number_of_edges()==0: #直到没有边了
                break            
            L = sorted(self.G1.nodes,key=lambda x:self.LNC(x),reverse=True)
            if L[0]>0:
                S.append(L[0])
            else:
                L=sorted(self.G1.nodes,key= self.G1.degree,reverse=True)
                S.append(L[0])

        S = {i:1 for i in S}
        self.G1=self.G.subgraph(self.G.nodes-S)
        while len(S) > k :            
            y = lambda x: self.f(S.keys()-{x} )
            L = sorted(S,key= y,reverse=False)
            S.pop(L[0])      

        result=self.f(S )
        print(S)
        print('f:',result)
        return result

    def LNC(self,i):        
        H = 0
        if not self.G1.has_node(i):
            return 0
        di = self.G1.degree(i) #度数
        for j in self.G1.neighbors(i):
            dj=self.G1.degree(j)
            if di+dj-2 !=0: #等于0说明只有两个点一个边，直接输出0
                i_j= set(self.G1.neighbors(i)) & set(self.G1.neighbors(j)) #i和j的共同邻居
                weigh = 1-   ( len(i_j) +1) / min(di,dj) #计算权重
                H += weigh * (di-1)/(di+dj-2)
            else:
                return 0
        return H


name='GCNP'
if __name__ == "__main__":
    Y = ["karate.txt","lesmis.gml"]  
    # Y = ["celegan.gml","Electronic_circuits.txt","yeast.txt","email.txt","polbooks.gml","karate.txt","lesmis.gml"]
    Z = [0.01,0.05,0.1,0.15,0.2,0.25,0.3]  #七个值 七个值
    result={name:{'karate.txt':()}}#存储结果

    for y in Y:
        # result[name][y]=list(0 for z in range(len(Z)))
        for z in range(len(Z)):     
            if y=="lesmis.gml" or y=="polbooks.gml" or y=="celegan.gml":
                pyl = PyLouvain.from_gml_file("C:\\Users\\xqjwo\\Desktop\\dataset\\"+y)
            else:
                pyl = PyLouvain.from_file("C:\\Users\\xqjwo\\Desktop\\dataset\\"+y)
            
            t0 = time.perf_counter()
            c = pyl.apply_method(Z[z]) #关键参数，是一个算法在一个网络的一个案例的结果
            print(c)


            













