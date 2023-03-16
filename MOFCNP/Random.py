#!/usr/bin/env python3
import copy
import networkx as nx
import numpy as np
import random
import math
import time
import sys

class PyLouvain:
    '''
        Builds a graph from _path.
        _path: a path to a file containing "node_from node_to" edges (one per line)
    '''
    @classmethod
    def from_file1(cls, path): #接收邻接矩阵
        f = open(path, 'r')
        lines = f.readlines()
        f.close()
        nodes = {}
        edges = []
        k=0 #记录当前行数
        for line in lines:
            nodes[k]=1 
            n = line.split()
            if not n:
                break
            m = len(n)  #一行有多少数
            for i in range(0,m):
                if int(n[i]) != 0:
                    nodes[i]=1
                    w=1  
                    if int(n[i]) != 1:
                        w = n[i]
                    edges.append(((k, i), w))
            k+=1
        # rebuild graph with successive identifiers
        # nodes_, edges_ = in_order(nodes, edges)
        print("%d nodes, %d edges" % (len(nodes), len(edges)))
        return cls(nodes, edges)

    @classmethod
    def from_file(cls, path):
        f = open(path, 'r')
        lines = f.readlines()
        f.close()
        nodes = {}
        edges = []
        for line in lines:
            n = line.split()
            if not n:
                break
            nodes[n[0]] = 1
            nodes[n[1]] = 1
            w = 1
            if len(n) == 3:
                w = int(n[2])
            edges.append(((n[0], n[1]), w))
        # rebuild graph with successive identifiers
        # nodes_, edges_ = in_order(nodes, edges)
        print("%d nodes, %d edges" % (len(nodes), len(edges)))
        return cls(nodes, edges)

    '''
        Builds a graph from _path.
        _path: a path to a file following the Graph Modeling Language specification
    '''
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
                edges.append(((current_edge[0], current_edge[1]), 1))
                current_edge = (-1, -1, 1)
                in_edge = 0
        # nodes, edges = in_order(nodes, edges)
        print("%d nodes, %d edges" % (len(nodes), len(edges)))
        return cls(nodes, edges)

    @classmethod
    def from_ERgraph(cls, n,m): # n为点数，m为边
        p = 2*m/(n*(n-1))
        G = nx.random_graphs.erdos_renyi_graph(n,p)
        Gnodes = {}
        Gedges = []
        for nd in G.nodes:
            Gnodes[nd]=1
        for ed in G.edges:
            Gedges.append(((ed[0],ed[1]),1))
        print("%d nodes, %d edges" % (len(Gnodes), len(Gedges)))
        return cls(Gnodes, Gedges)

    @classmethod
    def from_WSgraph(cls, a,b,p): # a为点数，b为邻居数，p为重连概率
        G = nx.random_graphs.watts_strogatz_graph(a,b,p)
        Gnodes = {}
        Gedges = []
        for nd in G.nodes:
            Gnodes[nd]=1
        for ed in G.edges:
            Gedges.append(((ed[0],ed[1]),1))
        print("%d nodes, %d edges" % (len(Gnodes), len(Gedges)))
        return cls(Gnodes, Gedges)

    @classmethod
    def from_BAgraph(cls, n,m): # n为点数，m为每次加入数
        G = nx.random_graphs.barabasi_albert_graph(n,m)
        Gnodes = {}
        Gedges = []
        for nd in G.nodes:
            Gnodes[nd]=1
        for ed in G.edges:
            Gedges.append(((ed[0],ed[1]),1))
        print("%d nodes, %d edges" % (len(Gnodes), len(Gedges)))
        return cls(Gnodes, Gedges)

    '''
        Initializes the method.
        _nodes: a list of ints
        _edges: a list of ((int, int), weight) pairs
    '''

    def __init__(self, G):
        self.G=G
        self.G1=self.G.copy()
        print('nodes:',self.G.number_of_nodes(), 'edges:',self.G.number_of_edges())

    # def __init__(self, nodes, edges):
    #     self.nodes = nodes
    #     self.edges = edges
    #     # precompute m (sum of the weights of all links in network)
    #     #            k_i (sum of the weights of the links incident to node i)
    #     self.m = 0
    #     self.k_i = [0 for n in nodes]
    #     self.edges_of_node = {}
    #     self.w = [0 for n in nodes]
    #     self.G = nx.Graph()
    #     self.G.add_nodes_from([n for n in nodes])  
    #     for e in edges:
    #         self.G.add_edge(e[0][0],e[0][1])
    #         # save edges by node
    #         if e[0][0] not in self.edges_of_node:
    #             self.edges_of_node[e[0][0]] = [e]
    #         else:
    #             self.edges_of_node[e[0][0]].append(e)
    #         if e[0][1] not in self.edges_of_node:
    #             self.edges_of_node[e[0][1]] = [e]
    #         elif e[0][0] != e[0][1]:
    #             self.edges_of_node[e[0][1]].append(e)
    #     self.e_o_n = copy.deepcopy(self.edges_of_node)
    #     self.nb={node:set(self.G.neighbors(node)) for node in nodes}

    
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
        #1 建立覆盖集
        S=random.sample(self.G.nodes,k)
        gn=self.G.nodes
        print(len(S))
        return self.f(S)

name='random'
if __name__ == "__main__":

    Y = ["celegan.gml","Electronic_circuits.txt","yeast.txt","email.txt","polbooks.gml","karate.txt","lesmis.gml"]
    Z = [0.01,0.05,0.1,0.15,0.2,0.25,0.3]  #七个值 七个值
    result={name:{'karate.txt':()}}#存储结果

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
            f = open("C:\\Users\\xqjwo\\Desktop\\dataset\\experiment_data1\\"+name+"2.txt", "a+")
            f.write(L+'\n')
            f.close()