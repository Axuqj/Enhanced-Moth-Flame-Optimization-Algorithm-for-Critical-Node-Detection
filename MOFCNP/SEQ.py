
import copy, math
#!/usr/bin/env python3
import copy
from os import remove, system
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
        Vertices=self.G.number_of_nodes()
        k_bound = K
        edge = self.G.edges

        graph = [   list(self.G.neighbors(i)) for i in range(Vertices) ]
    
        # Representing Equation 1
        def func(concom): 
            return concom*(concom - 1) / 2

        # Removes vertex v from graph G
        def remove_vertex(graph, v_star):
            nodes = graph[v_star]
            for u in nodes:
                graph[u].remove(v_star)
            graph[v_star] = []

        # Appendix
        def evaluate(v, visited, ap, parent, low, disc, time, subt_size, impact, cut_size):
            children = 0
            visited[v] = True
            disc[v] = time
            low[v] = time
            time += 1

            for u in graph[v]:
                if visited[u] == False:
                    parent[u] = v
                    children += 1
                    subt_size[u] = 1
                    impact[u] = 0
                    evaluate(u, visited, ap, parent, low, disc, time, subt_size, impact, cut_size)
                    # Check if the subtree rooted with u has a connection to 
                    # one of the ancestors of v
                    low[v] = min(low[v], low[u])
                    # (1) v IS root of DFS tree and has two or more chilren.
                    if parent[v] == -1 and children > 1:
                        ap[v] = True
                    #(2) If v IS NOT root and low value of one of its child is more 
                    # than discovery value of v. 
                    if parent[v] != -1 and low[u] >= disc[v]:
                        ap[v] = True
                        cut_size[v] += subt_size[u]
                        impact[v] += func(subt_size[u])
                # Update low value of u for parent function calls 
                elif u != parent[v]:
                    low[v] = min(low[v], disc[u])

        # ap == articulation point
        def art_point():
            removed = []
            visited = [False]*Vertices
            disc = [0]*Vertices
            low = [0]*Vertices
            cut_size = [0]*Vertices
            subt_size = [0]*Vertices
            impact = [0]*Vertices
            parent = [-1]*Vertices 
            ap = [False]*Vertices
            time = 0

            for node in range(Vertices):
                if visited[node] == False:
                    evaluate(node, visited, ap, parent, low, disc, time, subt_size, impact,cut_size)
            # Removes the APs
            print(ap,'\n')
            for index, value in enumerate(ap):                
                if len(removed) < k_bound and value == True:
                    remove_vertex(graph, index)
                    removed.append(index)
                    # print(len(removed))
                    # print(1)
                    
            for v, check in enumerate(visited):
                if check:
                    if ap[v]:
                        impact[v] += func(time - cut_size[v])
                    else:
                        impact[v] += func(time - 1)
            # print(removed, ap)
            # print(len(removed))
            print(removed)
            return removed
        
        return self.f(art_point())

if __name__ == "__main__":
    Y=['WattsStrogatz_n250.txt','BarabasiAlbert_n500m1.txt','ErdosRenyi_n250.txt']
    Z=[70,50,50]
    i=0
    pyl=PyLouvain.from_adjlist("C:\\Users\\xqjwo\\Desktop\\dataset\\新建文件夹\\ACO_CNP-master\\instancs\\"+Y[i])
    print(pyl.apply_method(Z[i]))


