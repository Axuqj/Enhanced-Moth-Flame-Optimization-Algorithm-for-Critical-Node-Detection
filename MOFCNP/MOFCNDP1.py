# Moth-flame optimization algorithm
import random as rd
from math import exp, cos, pi
from copy import deepcopy
import sys
import numpy as np
import operator
import time
import networkx as nx
import math

class PyMoF:
    def __init__(self,G,K):
        self.g=G
        self.K=K
        self.L=sorted(self.g.nodes, key=self.g.degree,reverse=True)

    def ini(self,SearchAgents_no,dim,ub,lb,pwp):
        population, fitness = [], []
        if isinstance(lb,int):  #判断下界是同一个数字，还是一个数列；下界相同，均为同一个数
            for i in range(SearchAgents_no):
                moth = []
                for j in range(dim):
                    moth.append(round(rd.uniform(lb, ub)))
                population.append(moth)

            #反向学习
            population_op= [ [ ub+lb-qq     for qq in pp]  for pp in population ] #取相反种群
            for pp in population_op:
                population.append(pp)            
            pfitness = self.getFitness(population)
            population,pfitness=   self.getFlame(population, pfitness, SearchAgents_no)

            S=self.L[0:self.K] #度数
            population[0]=[i for i in S]  
            bc= nx.centrality.betweenness_centrality(self.g,normalized=False)#介数
            bcl =sorted(bc.items(), key=operator.itemgetter(1),reverse = True)
            S=[i[0]   for i in bcl[0:self.K]] 
            population[1]=[i for i in S]

            #补充一个不错的初始解
            S=[]
            while 1: 
                G1=self.g.subgraph(self.g.nodes-S)
                if G1.number_of_edges()==0: #直到没有边了
                    break            
                L1 = sorted(G1.nodes,key=G1.degree,reverse=True)
                S.append(L1[0])
            S = {i:1 for i in S}
            while len(S) > self.K :
                    y = lambda x: self.fobj(S.keys()-{x})
                    L1 = sorted(S,key= y)
                    S.pop(L1[0])
        
            population[3]=[i for i in S]
            print('tanlan:',self.fobj(population[3]))

            node_num= self.g.number_of_nodes()
            K=self.K
            # pwp1=pwp
            pwp1=0.5  #0.5
            # k1=math.ceil(K*pwp1+K)
            # k_1=math.ceil(K-K*pwp1)
            kk = math.ceil(K/2)
            k_1=math.ceil(K-kk)
            k1=math.ceil(K+kk) 
            N=0
            Nc=10 #迭代次数
            fmin= (node_num-1)*node_num/2
            S={i:1 for i in S}
            S1=deepcopy(S)
            t1=time.time()
            t2=0
            while N < Nc: 
                while len(S1) < k1 : # 增
                    y = lambda x: self.fobj(S1.keys() | {x} )
                    L = sorted(self.g.nodes-S1.keys(),key= y)
                    S1[L[0]]=1

                while len(S1) >K :  #删
                    y = lambda x: self.fobj(S1.keys()-{x} )
                    L = sorted(S1.keys(),key= y) 
                    S1.pop(L[0])
                
                N+=1
                fnew=self.fobj(S1)
                if fnew>fmin: #保证结果有进步，否则回撤操作
                    S1=S
                else:
                    S=S1
                    fmin=fnew
                if fmin!=fnew:
                    t2=time.time()-t1
                else:t2=0

                while len(S1) > k_1 : #删
                    y = lambda x: self.fobj(S1.keys()-{x} )
                    L = sorted(S1.keys(),key= y)
                    S1.pop(L[0])

                while len(S1) < K :  #增
                    # y=rd.choice(list(self.g.nodes-S1.keys()))
                    # S1[y]=1
                    y = lambda x: self.fobj(S1.keys() | {x} )
                    L = sorted(self.g.nodes-S1.keys(),key= y)
                    S1[L[0]]=1

                N += 1 
                fnew=self.fobj(S1)
                if fnew>fmin: #保证结果有进步，否则回撤操作
                    S1=S
                else:
                    S=S1
                    fmin=fnew

            population[2]=[i for i in S]
            print('greedy:',self.fobj(population[2]))

            # 扰动两点互换
            ite_m=30            
            pw= self.K*pwp
            index=0 #popolation的下标
            for po in population: 
                ite=0
                if index < 3:continue
                while ite < ite_m:
                    ite+=1
                    for t in range(30):
                        newp=deepcopy(po) 
                        new_node=rd.choice(list(self.g.nodes-   newp))
                        newp.append(new_node)
                        L = sorted(newp,key=lambda x: self.fobj(set(newp)-{x} )) 
                        newp.remove(L[0])
                        if self.fobj(newp) < self.fobj(po): #保证结果有进步，否则回撤操作
                            population[index]=newp
                        else:
                            newp=population[index]
                    
                    pcount=0
                    while pcount<pw:
                        new_node=rd.choice(list(newp))
                        newp.remove(new_node)
                        pcount+=1
                    pcount=0
                    while pcount<pw:
                        new_node=rd.choice(list(self.g.nodes-newp))
                        newp.append(new_node)    
                        pcount+=1
                index+=1
        else:
            for i in range(SearchAgents_no):
                moth = []
                for j in range(dim):
                    moth.append(round(rd.uniform(lb[j], ub[j])))
                population.append(moth)
        return population,t2

    def fobj(self,s): #计算graph中除去s后，的连通对数.所有s都排除
        s={int(i) :1  for i in s}  #向下取整。比如100个点，0-100，只取得到0-99.
        G1=self.g.subgraph(self.g.nodes-s)
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

    def getFitness(self,moths):
        fitness = []
        for i in range(len(moths)):
            fitness.append(self.fobj(moths[i]))
        return fitness

    def getFlame(self,mothPopulation, mothFitness, flameNumber): #机制证实有效
        flamePopulation, flameFitness = [], []
        fitness = deepcopy(mothFitness)
        fitness.sort()
        for i in range(flameNumber):
            flameFitness.append(fitness[i])
            flamePopulation.append(mothPopulation[mothFitness.index(fitness[i])])
        return flamePopulation, flameFitness

    def check(self,x,ub,lb,no):
        if int(x) in no:
            for i in self.L:
                if i not in no:
                    return i   
        elif x < lb:
            return lb
        elif x > ub:
            return ub
        else:
            return x

    def MFO(self,N,Max_iteration,lb,ub,dim,  t_max,pwp):
        
        print('MFO is optimizing your problem')
        Convergence_curve=[0]*Max_iteration
        b = 1
        Moth_pos,t2 = self.ini(N, dim,ub,lb,pwp) #初始化

        previous_population=[]
        previous_population= deepcopy(Moth_pos)
        Best_flame_score=0
        Best_flame_pos=[]
        iterx= 0
        time_count=0
        # noo=[i for i in self.g.nodes if self.g.degree(i)==1 or self.g.degree(i)==0] #度为0/1的点去掉
        t0=time.time() #开始计时
        while time.time()-t0 < t_max:
            # a linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
            a=-1+(time.time()-t0)*((-1)/t_max)

            Flame_no=round(N-     (time.time()-t0)  *((N-1)/t_max    )) #四舍五入，火苗数量，随迭代次数，从N减到1       
            for i in range(N): #检查飞蛾:1.是否出界,2.是否与重复
                for j in range(dim):
                    no= [ int(Moth_pos[i][q])  for q in range(dim) if q <j ]     
                    Moth_pos[i][j]=self.check(Moth_pos[i][j],ub,lb,no)
                
            if iterx==0:
                mothFitness = self.getFitness(Moth_pos)
                flamePopulation, flameFitness = self.getFlame(Moth_pos, mothFitness, Flame_no)
            else:
                double_population=best_flames+previous_population
                mothFitness = self.getFitness(double_population)
                flamePopulation, flameFitness = self.getFlame(double_population, mothFitness, Flame_no)
            # Update the flames
            best_flames=deepcopy(flamePopulation)
            # best_flame_fitness=fitness_sorted
            # Update the position best flame obtained so far
            if Best_flame_score != flameFitness[0]:
                time_count= time.time()-t0   #获取最好成绩花的时间
            Best_flame_score= flameFitness[0]
            Best_flame_pos=best_flames[0]

            #更新moth
            for i in range(N):
                for j in range(dim):
                    b=1
                    t = rd.uniform(a, 1)
                    if i<Flame_no: # Update the position of the moth with respect to its corresponsing flame
                        distance_to_flame=abs(best_flames[i][j] - Moth_pos[i][j])                        
                        Moth_pos[i][j]=distance_to_flame*exp(b*t)*cos(t*2*pi)+best_flames[i][j]  
                    if i>=Flame_no: # Upaate the position of the moth with respct to one flame
                        distance_to_flame=abs(best_flames[0][j] - Moth_pos[i][j])
                        Moth_pos[i][j]=distance_to_flame*exp(b*t)*cos(t*2*pi)+best_flames[0][j]                   

         

                node_num= self.g.number_of_nodes()
                K=self.K
                k1=math.ceil(K*pwp+K)
                k_1=math.ceil(K-K*pwp)
                N=0
                Nc=3 #迭代次数
                fmin= (node_num-1)*node_num/2
                S={i:1 for i in Moth_pos[i]}
                S1=deepcopy(S)
                while N < Nc: 
                    while len(S1) < k1 : # 增
                        # y = lambda x: self.fobj(S1.keys() | {x} )
                        # L = sorted(self.g.nodes-S1.keys(),key= y)
                        y=rd.choice(list(self.g.nodes-S1.keys()))
                        S1[y]=1

                    while len(S1) >K :  #删
                        y = lambda x: self.fobj(S1.keys()-{x} )
                        L = sorted(S1.keys(),key= y) 
                        S1.pop(L[0])
                    
                    N+=1
                    fnew=self.fobj(S1)
                    if fnew>fmin: #保证结果有进步，否则回撤操作
                        S1=S
                    else:
                        S=S1
                        fmin=fnew

                    while len(S1) > k_1 : #删
                        y = lambda x: self.fobj(S1.keys()-{x} )
                        L = sorted(S1.keys(),key= y)
                        S1.pop(L[0])

                    while len(S1) < K :  #增
                        # y = lambda x: self.fobj(S1.keys() | {x} )
                        # L = sorted(self.g.nodes-S1.keys(),key= y)
                        y=rd.choice(list(self.g.nodes-S1.keys()))
                        S1[y]=1
                                    
                    N += 1 
                    fnew=self.fobj(S1)
                    if fnew>fmin: #保证结果有进步，否则回撤操作
                        S1=S
                    else:
                        S=S1
                        fmin=fnew
                Moth_pos[i]=[i for i in S]
            previous_population=Moth_pos

            # Convergence_curve[iterx]=Best_flame_score        
            # Display the iteration and best optimum obtained so far
            if iterx % (Max_iteration/10)==0:
                print('At iteration '+str(iterx)+ ' the best fitness is '+ str(Best_flame_score))
            iterx += 1
        print('spent time :', round(time_count))
        return Best_flame_score,Best_flame_pos ,Convergence_curve, round(time_count+t2)

    def MFOCNP1(self,SearchAgents_no=10,  t_max=300,pwp=0.2):
        # SearchAgents_no=30
        # Function_name='F1'
        # Max_iteration=1000
        # lb,ub,dim,fobj=Get_Functions_details(Function_name)
        Max_iteration=1000
        lb=0
        ub=self.g.number_of_nodes()-1.0e-10 # 0-n
        dim=self.K
        Best_score,Best_pos,cg_curve,t2=self.MFO(SearchAgents_no,Max_iteration,lb,ub,dim,  t_max,pwp)
        Best_pos=[int(i) for i in Best_pos]
        print(Best_score)
        print(Best_pos)
        return Best_score,Best_pos,t2

if __name__ == '__main__':
    SearchAgents_no=30
    # Function_name='F1'
    # Max_iteration=1000
    # # lb,ub,dim,fobj=Get_Functions_details(Function_name)
    # lb=-100
    # ub=100
    # dim=10
    # Best_score,Best_pos,cg_curve=MFO(SearchAgents_no,Max_iteration,lb,ub,dim)
    # print(Best_score,Best_pos)
