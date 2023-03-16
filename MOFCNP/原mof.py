# Moth-flame optimization algorithm
import random as rd
from math import exp, cos, pi
from copy import deepcopy
import sys
import numpy as np

def ini(SearchAgents_no,dim,ub,lb):
    population, fitness = [], []
    if isinstance(lb,int):  #判断下界是同一个数字，还是一个数列；下界相同，均为同一个数
        for i in range(SearchAgents_no):
            moth = []
            for j in range(dim):
                moth.append(round(rd.uniform(lb, ub)))
            population.append(moth)
    else:
        for i in range(SearchAgents_no):
            moth = []
            for j in range(dim):
                moth.append(round(rd.uniform(lb[j], ub[j])))
            population.append(moth)
    return population

def fobj(moth):
    objFunctionValue = 0

    # for i in range(len(moth)):
    #     objFunctionValue += moth[i] ** 2

    for i in range(len(moth)):
        objFunctionValue+= sum(moth[0:i])**2
    return objFunctionValue

def getFitness(moths):
    fitness = []
    for i in range(len(moths)):
        fitness.append(fobj(moths[i]))
    return fitness

def getFlame(mothPopulation, mothFitness, flameNumber):
    flamePopulation, flameFitness = [], []
    fitness = deepcopy(mothFitness)
    fitness.sort()
    for i in range(flameNumber):
        flameFitness.append(fitness[i])
        flamePopulation.append(mothPopulation[mothFitness.index(fitness[i])])
    return flamePopulation, flameFitness

# def Get_Functions_details(F):
#     if F=='F1':
#         fobj = lambda x:sum([y for y in x])
#         # fobj = lambda x:sum([y**2 for y in x])
#         lb=-100
#         ub=100
#         dim=5
#         return lb,ub,dim,fobj

def check(x,ub,lb):
    if x < lb:
        return round(lb)
    elif x > ub:
        return round(ub)
    else:
        return x

def MFO(N,Max_iteration,lb,ub,dim):
    print('MFO is optimizing your problem')
    Convergence_curve=[0]*Max_iteration
    b = 1
    Moth_pos = ini(N, dim,ub,lb) #初始化
    previous_population=Moth_pos 
    best_flames=[]
    iterx= 0
    while iterx < Max_iteration:
        Flame_no=round(N-iterx*((N-1)/Max_iteration))
        for i in range(N):
            for j in range(dim):
                check(Moth_pos[i][j],ub,lb)
        
        #Sort the moths 
        if iterx==0:
            mothFitness = getFitness(Moth_pos)
            flamePopulation, flameFitness = getFlame(Moth_pos, mothFitness, Flame_no)
        else:
            double_population=previous_population+best_flames
            mothFitness = getFitness(double_population)
            flamePopulation, flameFitness = getFlame(double_population, mothFitness, Flame_no)
        
        # Update the flames
        best_flames=flamePopulation
        # best_flame_fitness=fitness_sorted
        # Update the position best flame obtained so far
        Best_flame_score= flameFitness[0]
        Best_flame_pos=best_flames[0]
        
        # a linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
        a=-1+iterx*((-1)/Max_iteration)

        #更新moth
        for i in range(N):
            for j in range(dim):
                if i<Flame_no: # Update the position of the moth with respect to its corresponsing flame
                    distance_to_flame=abs(best_flames[i][j] - Moth_pos[i][j])
                    b=1
                    t = rd.uniform(a, 1)
                    Moth_pos[i][j]=distance_to_flame*exp(b*t)*cos(t*2*pi)+best_flames[i][j]              
                if i>=Flame_no: # Upaate the position of the moth with respct to one flame
                    distance_to_flame=abs(best_flames[Flame_no-1][j] - Moth_pos[i][j])
                    b=1
                    t = rd.uniform(a, 1)
                    Moth_pos[i][j]=distance_to_flame*exp(b*t)*cos(t*2*pi)+best_flames[Flame_no-1][j]
        previous_population=Moth_pos
        Convergence_curve[iterx]=Best_flame_score
    
        # Display the iteration and best optimum obtained so far
        if iterx % (Max_iteration/10)==0:
            print('At iteration '+str(iterx)+ ' the best fitness is '+ str(Best_flame_score))
        iterx += 1
    return Best_flame_score,Best_flame_pos,Convergence_curve

if __name__ == '__main__':
    SearchAgents_no=30
    Function_name='F1'
    Max_iteration=1000
    # lb,ub,dim,fobj=Get_Functions_details(Function_name)
    lb=-100
    ub=100
    dim=10
    Best_score,Best_pos,cg_curve=MFO(SearchAgents_no,Max_iteration,lb,ub,dim)
    print(Best_score,Best_pos)
