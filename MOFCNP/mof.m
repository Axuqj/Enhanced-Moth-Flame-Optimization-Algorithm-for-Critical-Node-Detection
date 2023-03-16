# Moth-flame optimization algorithm
import random as rd
from math import exp, cos, pi
from copy import deepcopy

def ini(n, d):
    population, fitness = [], []
    for i in range(n):
        moth = []
        for j in range(d):
            moth.append(rd.uniform(-100, 100))
        population.append(moth)
    return population

def getFitness(moths):
    fitness = []
    for i in range(len(moths)):
        fitness.append(objFunction(moths[i]))
    return fitness

def objFunction(moth):
    objFunctionValue = 0
    for i in range(len(moth)):
        objFunctionValue += moth[i] ** 2
    return objFunctionValue

def run():
    number, dimension = 30, 10
    b = 1
    mothPopulation = ini(number, dimension)
    iterx, maxIterx = 0, 1000
    while iterx < maxIterx:
        mothFitness = getFitness(mothPopulation)
        if iterx > 90:
            flameNumber = 1
        elif iterx == 0:
            flameNumber = 5
        else:
            flameNumber = int((maxIterx - iterx) / 200)  +1
        flamePopulation, flameFitness = getFlame(mothPopulation, mothFitness, flameNumber)

        for i in range(number):
            for j in range(dimension):
                r = -1 - 0.01 * iterx
                t = rd.uniform(r, 1)
                if i < len(flamePopulation):
                    distance = abs(flamePopulation[i][j] - mothPopulation[i][j])
                    mothPopulation[i][j] = distance * exp(b * t) * cos(2 * pi * t) + flamePopulation[i][j]
                    mothPopulation[i][j] = check(mothPopulation[i][j])
                else:
                    distance = abs(flamePopulation[0][j] - mothPopulation[i][j])
                    mothPopulation[i][j] = distance * exp(b * t) * cos(2 * pi * t) + flamePopulation[0][j]
                    mothPopulation[i][j] = check(mothPopulation[i][j])
        iterx += 1
    
    print(flameFitness,flamePopulation  )

def getFlame(mothPopulation, mothFitness, flameNumber):
    flamePopulation, flameFitness = [], []
    fitness = deepcopy(mothFitness)
    fitness.sort()
    for i in range(flameNumber):
        flameFitness.append(fitness[i])
        flamePopulation.append(mothPopulation[mothFitness.index(fitness[i])])
    return flamePopulation, flameFitness

def check(x):
    if x < -100:
        return -100
    elif x > 100:
        return 100
    else:
        return x

if __name__ == '__main__':
    run()


