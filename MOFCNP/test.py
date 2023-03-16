import copy
from os import system
import networkx as nx
import numpy as np
import math
import time
import operator
import sys
import random 
import time
import copy 
import networkx as nx

def from_file(path):
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    G = nx.Graph()
    for line in lines:
        n = line.split()
        if not n:
            break
        G.add_edge(int(n[0]), int(n[1]))
    print( G.number_of_nodes(), G.number_of_edges())
    return G

def from_gml_file( path):
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    G = nx.Graph()
    current_edge = (-1, -1)
    in_edge = 0
    for line in lines:
        words = line.split()
        if not words:
            break
        if words[0] == 'source':
            current_edge = (int(words[1]), current_edge[1])
        elif words[0] == 'target':
            G.add_edge(current_edge[0],int(words[1]))
    print( G.number_of_nodes(), G.number_of_edges())
    return G


Y = ["celegan.gml","Electronic_circuits.txt","yeast.txt","email.txt","polbooks.gml","karate.txt","lesmis.gml"]

for y in Y:
    if y=="lesmis.gml" or y=="polbooks.gml" or y=="celegan.gml":
        G= from_gml_file('C:\\Users\\xqjwo\\Desktop\\dataset\\'+y)
    else: G= from_file('C:\\Users\\xqjwo\\Desktop\\dataset\\'+y)