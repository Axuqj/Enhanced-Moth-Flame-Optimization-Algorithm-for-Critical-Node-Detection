#!/usr/bin/env python3
from communities.algorithms import louvain_method
import copy
import networkx as nx
import numpy 
import gc
import operator
import random
import time
import math
import sys
from communities.algorithms import louvain_method
from communities.visualization import draw_communities as draw

class PyLouvain:
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
    def from_gml_file(cls, path):
        f = open(path, 'r')
        lines = f.readlines()
        f.close()
        nodes = {}
        edges = []
        current_edge = (-1, -1, 1)
        in_edge = 0
        for line in lines:
            words = line.split()
            if not words:
                break
            if words[0] == 'id':
                nodes[int(words[1])] = 1
            elif words[0] == 'source':
                in_edge = 1
                current_edge = (int(words[1]), current_edge[1], current_edge[2])
            elif words[0] == 'target' and in_edge:
                current_edge = (current_edge[0], int(words[1]), current_edge[2])
            elif words[0] == 'value' and in_edge:
                current_edge = (current_edge[0], current_edge[1], int(words[1]))
            elif words[0] == ']' and in_edge:
                edges.append(((int(current_edge[0]), int(current_edge[1])), 1))
                current_edge = (-1, -1, 1)
                in_edge = 0
        return cls(nodes, edges)

    @classmethod
    def from_ERgraph(cls, n,m): # n为点数，m为边
        p = 2*m/(n*(n-1))
        G = nx.random_graphs.erdos_renyi_graph(n,p)
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

    '''
        Applies the Louvain method.
    '''
    def apply_method(self,z):  #社区挖掘
        adj_matrix = numpy.array(nx.to_numpy_matrix(self.G))
        actual_partition, _=louvain_method(adj_matrix)
        partition=actual_partition
        network = (self.G.nodes, self.G.edges)    

        k = math.ceil(len(self.G.nodes)*z)  #参数为0.2
        #得到actual_partition，为社区挖掘结果
        # print(best_q)
        # return 0
        #第二步 提取初始覆盖集
        S={}    

        # 构建中心性指标K
        kpool=[]
        K={node : 0 for node in network[0]}  
        for node in range(len(network[0])): #n为排序序数
            d=self.G.degree(node)
            if d != 0 and d != 1: #度数为0或1的不参与
                K[node]=d
                kpool+=[node]*d
        # K={node : 0 for node in network[0]}  #指标由度数和局部介数指数组成
        # for partition in actual_partition:
        #     cluster=self.G.subgraph(partition).copy()
        #     bc1= nx.centrality.betweenness_centrality(cluster,normalized=False)
        #     for b in bc1.keys():
        #         if bc1[b]*len(bc1)>30: K[b]=self.G.degree(b)+30
        #         elif bc1[b]*len(bc1)>100: K[b]=self.G.degree(b)+50
        #         elif bc1[b]*len(bc1)>100: K[b]=self.G.degree(b)+70
        #         else: K[b]=self.G.degree(b)+bc1[b]*len(bc1)
        #         kpool+=[b]*math.ceil(K[b])


        # for p in actual_partition:
        #     for q in p:
        #         for i in self.G.neighbors(q):
        #             if i not in p:
        #                 S[q],S[i]=1,1

        # #筛选每个社区内，hub节点
        # hub=[]
        # for partition in actual_partition:
        #     cluster=self.G.subgraph(partition).copy()
        #     c_n=math.ceil(len(cluster)*0.3)
        #     bc1= nx.centrality.betweenness_centrality(cluster,normalized=False)
        #     bc2 = sorted(bc1.items(), key=operator.itemgetter(1),reverse = True)
        #     for bcl1 in bc2:
        #         if bcl1[0] not in hub:
        #             hub.append(bcl1[0])
        #             c_n -= 1
        #             if c_n == 0:
        #                 break
        # for bc3 in hub:
        #     S[bc3]=1

        S=[]
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
        S2 = {i:1 for i in S}
        k1=k/3
        kk=math.ceil(k)
        k0=math.ceil(k-k1)
        k2=math.ceil(k+k1)   
        # print(kk,k0,k2)
        while len(S) > kk:  #每次删除一个点
            y = lambda x: self.f(S.keys()-{x} )
            L = sorted(S.keys(),key= y)
            S.pop(L[0])
        S2= {i:1 for i in S}


        Nk=60 #迭代次数
        nk=0
    
        while nk<Nk:
            # print(len(S))
            while len(S2) > k0:  #每次删一个点
                y = lambda x: self.f(S2.keys()-{x} )
                L = sorted(S2.keys(),key= y)
                S2.pop(L[0])
            
            kpool1=[i for i in kpool if i not in S2]
            while len(S2) <  kk:
                noder= numpy.random.choice(kpool1)
                S2[noder]=1 

            if self.f(S2 )>self.f(S ): #保证结果有进步，否则回撤操作
                S2={i:1 for i in S}
            else:
                S={i:1 for i in S2}
        
            nk+=1

            kpool1=[i for i in kpool if i not in S2]
            while len(S2) < k2:  #继续添加，每次添加一个点
                noder= numpy.random.choice(kpool1)
                S2[noder]=1 #选出最大的，加入S

            while len(S2) > kk:  #每次删除一个点
                y = lambda x: self.f(S2.keys()-{x} )
                L = sorted(S2.keys(),key= y)
                # if k/10 >=10:
                #     for i in range(7):
                #         S2.pop(L[i])
                #         if len(S2) <= kk:
                #             break
                # else:
                S2.pop(L[0])

            if self.f(S2 )>self.f(S ): #保证结果有进步，否则回撤操作
                S2={i:1 for i in S}
            else:
                S={i:1 for i in S2}

            nk+=1  #n=2
        result=self.f(S)
        print(result)
        return result


def realworld():
    # Y = ["celegan.gml","Electronic_circuits.txt","yeast.txt","email.txt","polbooks.gml","karate.txt","lesmis.gml"]
    Y=["lesmis.gml","karate.txt","lesmis.gml"]
    # Z = [0.01,0.05,0.1,0.15,0.2,0.25,0.3]  #七个值 七个值
    Z =[0.01,0.1,0.3] 
    result={name:{"lesmis.gml":()}}#存储结果
    
    f = open("C:\\Users\\xqjwo\\Desktop\\dataset\\experiment_data1\\test.txt", "a+")
    f.write(name+'\n')
    f.close()

    for y in Y:
        result[name][y]=list(0 for z in range(len(Z)))
        for z in range(len(Z)):     
            if y=="lesmis.gml" or y=="polbooks.gml" or y=="celegan.gml":
                pyl = PyLouvain.from_gml_file("C:\\Users\\xqjwo\\Desktop\\dataset\\"+y)
            else:
                pyl = PyLouvain.from_file("C:\\Users\\xqjwo\\Desktop\\dataset\\"+y)
            
            t0 = time.perf_counter()
            c = pyl.apply_method(Z[z]) #关键参数，是一个算法在一个网络的一个案例的结果
            #连通节点对 表示算法name在网络y的第i个参数的结果是c：result={'CBG':{'network1': (1,2,3), '':()}, ''   }
            result[name][y][z]=c
            L = name+' '+y+'参数'+str(z)+' '+str(int(c))+' '+str(int(time.perf_counter() - t0))
            f = open("C:\\Users\\xqjwo\\Desktop\\dataset\\experiment_data1\\test.txt", "a+")
            f.write(L+'\n')
            f.close()

def ER_BA():
    # WS
    Z=['250 1250 70','  500 1500 125',' 1000 5000 200',' 1500 4500 265']
    for z in Z:
        z=z.split()
        t0 = time.perf_counter()
        n = 2*int(z[1])/int(z[0])
        pyl = PyLouvain.from_WSgraph(int(z[0]),int(n),0.3)
        c = pyl.apply_method(int(z[2])/len(pyl.nodes))
        L = name+' '+'WS'+str(z[0])+str(z[2])+' '+str(int(c))+' '+str(int(time.perf_counter() - t0))
        # f = open("C:\\Users\\xqjwo\\Desktop\\dataset\\experiment_data\\test.txt", "a+")
        f = open("C:\\Users\\xqjwo\\Desktop\\dataset\\experiment_data\\test.txt", "a+")
        f.write(L+'\n')
        f.close()
        print(L)
    #BA
    Z=['500 499 50',' 1000 999 75',' 2500 2499 100','5000 4999 150']
    for z in Z:
        z=z.split()
        t0 = time.perf_counter()
        pyl = PyLouvain.from_BAgraph(int(z[0]),1)
        c = pyl.apply_method(int(z[2])/len(pyl.nodes))
        L = name+' '+'BA'+str(z[0])+str(z[2])+' '+str(int(c))+' '+str(int(time.perf_counter() - t0))
        # f = open("C:\\Users\\xqjwo\\Desktop\\dataset\\experiment_data\\test.txt", "a+")
        f = open("C:\\Users\\xqjwo\\Desktop\\dataset\\experiment_data\\test.txt", "a+")
        f.write(L+'\n')
        f.close()
        print(L)
    # ER
    Z=['235 349 50','465 699 80',' 940 1399 140','2343 3499 200']
    for z in Z:
        z=z.split()
        t0 = time.perf_counter()
        pyl = PyLouvain.from_ERgraph(int(z[0]),int(z[1]))
        c = pyl.apply_method(int(z[2])/len(pyl.nodes))
        L = name+' '+'ER'+str(z[0])+str(z[2])+' '+str(int(c))+' '+str(int(time.perf_counter() - t0))
        f = open("C:\\Users\\xqjwo\\Desktop\\dataset\\experiment_data\\test.txt", "a+")
        f.write(L+'\n')
        f.close()
        print(L)

name='TBH'
if __name__ == "__main__":
    Y=['WattsStrogatz_n250.txt','BarabasiAlbert_n500m1.txt','ErdosRenyi_n250.txt']
    Z=[70,50,50]
    for i in range(3):
        pyl = PyLouvain.from_file("C:\\Users\\xqjwo\\Desktop\\dataset\\标准数据集\\"+Y[i])
        print(pyl.apply_method(Z[i]))



