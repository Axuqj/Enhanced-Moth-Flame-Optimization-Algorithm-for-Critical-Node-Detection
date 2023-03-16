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


    def function1(self,x): # the number of connected components
        s={int(i) :1  for i in x}  #向下取整。比如100个点，0-100，只取得到0-99.
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
        return id


    #Second function to optimize
    def function2(self,x):
        G1=self.G.subgraph(self.G.nodes-x)
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

    #Function to find index of list
    def index_of(a,list):
        for i in range(0,len(list)):
            if list[i] == a:
                return i
        return -1

    #Function to sort by values
    def sort_by_values(self,list1, values):
        sorted_list = []
        while(len(sorted_list)!=len(list1)):
            if self.index_of(min(values),values) in list1:
                sorted_list.append(self.index_of(min(values),values))
            values[self.index_of(min(values),values)] = math.inf
        return sorted_list

    #Function to carry out NSGA-II's fast non dominated sort
    def fast_non_dominated_sort(self,values1, values2):
        S=[[] for i in range(0,len(values1))]
        front = [[]]
        n=[0 for i in range(0,len(values1))]
        rank = [0 for i in range(0, len(values1))]

        for p in range(0,len(values1)):
            S[p]=[]
            n[p]=0
            for q in range(0, len(values1)):
                if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                    if q not in S[p]:
                        S[p].append(q)
                elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                    n[p] = n[p] + 1
            if n[p]==0:
                rank[p] = 0
                if p not in front[0]:
                    front[0].append(p)

        i = 0
        while(front[i] != []):
            Q=[]
            for p in front[i]:
                for q in S[p]:
                    n[q] =n[q] - 1
                    if( n[q]==0):
                        rank[q]=i+1
                        if q not in Q:
                            Q.append(q)
            i = i+1
            front.append(Q)

        del front[len(front)-1]
        return front

    #Function to calculate crowding distance
    def crowding_distance(self,values1, values2, front):
        distance = [0 for i in range(0,len(front))]
        sorted1 = self.sort_by_values(front, values1[:])
        sorted2 = self.sort_by_values(front, values2[:])
        distance[0] = 4444444444444444
        distance[len(front) - 1] = 4444444444444444
        for k in range(1,len(front)-1):
            distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
        for k in range(1,len(front)-1):
            distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
        return distance

    #Function to carry out the crossover
    def crossover(self,a,b):
        r=random.random()
        if r>0.5:
            return self.mutation((a+b)/2)
        else:
            return self.mutation((a-b)/2)

    #Function to carry out the mutation operator
    def mutation(self,solution):
        mutation_prob = random.random()
        if mutation_prob <1:
            solution = self.min_x+(self.max_x-self.min_x)*random.random()
        return solution


    def apply_method(self,k):
        #Main program starts here
        pop_size = k
        max_gen = 921

        #Initialization
        min_x=0
        max_x=self.G.number_of_nodes()
        solution=[min_x+(max_x-min_x)*random.random() for i in range(0,pop_size)]


        gen_no=0
        while(gen_no<max_gen):
            function1_values = [self.function1(solution[i])for i in range(0,pop_size)]
            function2_values = [self.function2(solution[i])for i in range(0,pop_size)]
            non_dominated_sorted_solution = self.fast_non_dominated_sort(function1_values[:],function2_values[:])
            print("The best front for Generation number ",gen_no, " is")
            for valuez in non_dominated_sorted_solution[0]:
                print(round(solution[valuez],3),end=" ")
            print("\n")
            crowding_distance_values=[]
            for i in range(0,len(non_dominated_sorted_solution)):
                crowding_distance_values.append(self.crowding_distance(function1_values[:],function2_values[:],non_dominated_sorted_solution[i][:]))
            solution2 = solution[:]
            #Generating offsprings
            while(len(solution2)!=2*pop_size):
                a1 = random.randint(0,pop_size-1)
                b1 = random.randint(0,pop_size-1)
                solution2.append(self.crossover(solution[a1],solution[b1]))
            function1_values2 = [self.function1(solution2[i])for i in range(0,2*pop_size)]
            function2_values2 = [self.function2(solution2[i])for i in range(0,2*pop_size)]
            non_dominated_sorted_solution2 = self.fast_non_dominated_sort(function1_values2[:],function2_values2[:])
            crowding_distance_values2=[]
            for i in range(0,len(non_dominated_sorted_solution2)):
                crowding_distance_values2.append(self.crowding_distance(function1_values2[:],function2_values2[:],non_dominated_sorted_solution2[i][:]))
            new_solution= []
            for i in range(0,len(non_dominated_sorted_solution2)):
                non_dominated_sorted_solution2_1 = [self.index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]
                front22 = self.sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
                front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
                front.reverse()
                for value in front:
                    new_solution.append(value)
                    if(len(new_solution)==pop_size):
                        break
                if (len(new_solution) == pop_size):
                    break
            solution = [solution2[i] for i in new_solution]
            gen_no = gen_no + 1

        Solution=[]
        for i in range(len(solution)):
            if i==0:
                Solution=solution[i]
                continue
            if self.f(Solution)>self.f(solution[i]):Solution=solution[i]
        return self.f(Solution)

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