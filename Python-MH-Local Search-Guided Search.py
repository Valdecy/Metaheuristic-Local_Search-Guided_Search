############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Guided Search

# Citation: 
# PEREIRA, V. (2018). Project: Metaheuristic-Local_Search-Guided_Search, File: Python-MH-Local Search-Guided Search.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Local_Search-Guided_Search>

############################################################################

# Required Libraries
import pandas as pd
import numpy as np
import random
import copy
from matplotlib import pyplot as plt 

# Function: Tour Distance
def distance_calc(Xdata, city_tour):
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m = k + 1
        distance = distance + Xdata[city_tour[0][k]-1, city_tour[0][m]-1]            
    return distance

# Function: Euclidean Distance 
def euclidean_distance(x, y):       
    distance = 0
    for j in range(0, len(x)):
        distance = (x[j] - y[j])**2 + distance   
    return distance**(1/2) 

# Function: Initial Seed
def seed_function(Xdata):
    seed = [[],float("inf")]
    sequence = random.sample(list(range(1,Xdata.shape[0]+1)), Xdata.shape[0])
    sequence.append(sequence[0])
    seed[0] = sequence
    seed[1] = distance_calc(Xdata, seed)
    return seed

# Function: Build Distance Matrix
def buid_distance_matrix(coordinates):
    Xdata = np.zeros((coordinates.shape[0], coordinates.shape[0]))
    for i in range(0, Xdata.shape[0]):
        for j in range(0, Xdata.shape[1]):
            if (i != j):
                x = coordinates[i,:]
                y = coordinates[j,:]
                Xdata[i,j] = euclidean_distance(x, y)        
    return Xdata

# Function: Tour Plot
def plot_tour_distance_matrix (Xdata, city_tour):
    m = np.copy(Xdata)
    for i in range(0, Xdata.shape[0]):
        for j in range(0, Xdata.shape[1]):
            m[i,j] = (1/2)*(Xdata[0,j]**2 + Xdata[i,0]**2 - Xdata[i,j]**2)    
    w, u = np.linalg.eig(np.matmul(m.T, m))
    s = (np.diag(np.sort(w)[::-1]))**(1/2) 
    coordinates = np.matmul(u, s**(1/2))
    coordinates = coordinates.real[:,0:2]
    xy = np.zeros((len(city_tour[0]), 2))
    for i in range(0, len(city_tour[0])):
        if (i < len(city_tour[0])):
            xy[i, 0] = coordinates[city_tour[0][i]-1, 0]
            xy[i, 1] = coordinates[city_tour[0][i]-1, 1]
        else:
            xy[i, 0] = coordinates[city_tour[0][0]-1, 0]
            xy[i, 1] = coordinates[city_tour[0][0]-1, 1]
    plt.plot(xy[:,0], xy[:,1], marker = 's', alpha = 1, markersize = 7, color = 'black')
    plt.plot(xy[0,0], xy[0,1], marker = 's', alpha = 1, markersize = 7, color = 'red')
    plt.plot(xy[1,0], xy[1,1], marker = 's', alpha = 1, markersize = 7, color = 'orange')
    return

# Function: Tour Plot
def plot_tour_coordinates (coordinates, city_tour):
    xy = np.zeros((len(city_tour[0]), 2))
    for i in range(0, len(city_tour[0])):
        if (i < len(city_tour[0])):
            xy[i, 0] = coordinates[city_tour[0][i]-1, 0]
            xy[i, 1] = coordinates[city_tour[0][i]-1, 1]
        else:
            xy[i, 0] = coordinates[city_tour[0][0]-1, 0]
            xy[i, 1] = coordinates[city_tour[0][0]-1, 1]
    plt.plot(xy[:,0], xy[:,1], marker = 's', alpha = 1, markersize = 7, color = 'black')
    plt.plot(xy[0,0], xy[0,1], marker = 's', alpha = 1, markersize = 7, color = 'red')
    plt.plot(xy[1,0], xy[1,1], marker = 's', alpha = 1, markersize = 7, color = 'orange')
    return

# Function: Stochastic 2_opt
def stochastic_2_opt(Xdata, city_tour):
    best_route = copy.deepcopy(city_tour)      
    i, j  = random.sample(range(0, len(city_tour[0])-1), 2)
    if (i > j):
        i, j = j, i
    best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))           
    best_route[0][-1]  = best_route[0][0]              
    best_route[1] = distance_calc(Xdata, best_route)                     
    return best_route

#Function: Augmented Cost
def augumented_cost(Xdata, city_tour, penalty, limit):
    augmented = 0   
    for i in range(0, len(city_tour[0]) - 1):
        c1 = city_tour[0][i]
        c2 = city_tour[0][i + 1]      
        if c2 < c1:
            c1, c2 = c2, c1            
        augmented = augmented + Xdata[c1-1, c2-1] + (limit * penalty[c1-1][c2-1])    
    return augmented

# Function: Local Search
def local_search(Xdata, city_tour, penalty, max_attempts = 50, limit= 1):
    count = 0
    ag_cost = augumented_cost(Xdata, city_tour = city_tour, penalty = penalty, limit = limit)
    solution = copy.deepcopy(city_tour) 
    while (count < max_attempts):
        candidate = stochastic_2_opt(Xdata, city_tour = solution)
        candidate_augmented = augumented_cost(Xdata, city_tour = candidate, penalty = penalty, limit = limit)       
        if candidate_augmented < ag_cost:
            solution  = copy.deepcopy(candidate)
            ag_cost = augumented_cost(Xdata, city_tour = solution, penalty = penalty, limit = limit)
            count = 0
        else:
            count = count + 1                             
    return solution 

#Function: Utility
def utility (Xdata, city_tour, penalty, limit = 1):
    utilities = [0 for i in city_tour[0]]
    for i in range(0, len(city_tour[0]) - 1):
        c1 = city_tour[0][i]
        c2 = city_tour[0][i + 1]      
        if c2 < c1:
            c1, c2 = c2, c1            
        utilities[i] = Xdata[c1-1, c2-1] /(1 + penalty[c1-1][c2-1])  
    return utilities

#Function: Update Penalty
def update_penalty(penalty, city_tour, utilities):
    max_utility = max(utilities)   
    for i in range(0, len(city_tour[0]) - 1):
        c1 = city_tour[0][i]
        c2 = city_tour[0][i + 1]         
        if c2 < c1:
            c1, c2 = c2, c1        
        if (utilities[i] == max_utility):
            penalty[c1-1][c2-1] = penalty[c1-1][c2-1] + 1   
    return penalty

# Function: Guided Search
def guided_search(Xdata, city_tour, alpha = 0.3, local_search_optima = 1000, max_attempts = 20, iterations = 50):
    count = 0
    limit = alpha * (local_search_optima / len(city_tour[0]))  
    penalty = [[0 for i in city_tour[0]] for j in city_tour[0]]
    solution = copy.deepcopy(city_tour)
    best_solution = [[],float("inf")]
    while (count < iterations):
        solution = local_search(Xdata, city_tour = solution, penalty = penalty, max_attempts = max_attempts, limit = limit)
        utilities = utility(Xdata, city_tour = solution, penalty = penalty, limit = limit)
        penalty = update_penalty(penalty = penalty, city_tour = solution, utilities = utilities)
        if (solution[1] < best_solution[1]):
            best_solution = copy.deepcopy(solution) 
        count = count + 1
        print("Iteration = ", count, " Distance ", best_solution[1])
    return best_solution

######################## Part 1 - Usage ####################################

# Load File - A Distance Matrix (17 cities,  optimal = 1922.33)
X = pd.read_csv('Python-MH-Local Search-Guided Search-Dataset-01.txt', sep = '\t') 
X = X.values

# Start a Random Seed
seed = seed_function(X)

# Call the Function
lsgs = guided_search(X, city_tour = seed, alpha = 0.5, local_search_optima = 100, max_attempts = 10, iterations = 2500)

# Plot Solution. Red Point = Initial city; Orange Point = Second City # The generated coordinates (2D projection) are aproximated, depending on the data, the optimum tour may present crosses
plot_tour_distance_matrix(X, lsgs)

######################## Part 2 - Usage ####################################

# Load File - Coordinates (Berlin 52,  optimal = 7544.37)
Y = pd.read_csv('Python-MH-Local Search-Guided Search-Dataset-02.txt', sep = '\t') 
Y = Y.values

# Build the Distance Matrix
X = buid_distance_matrix(Y)

# Start a Random Seed
seed = seed_function(X)

# Call the Function
lsgs = guided_search(X, city_tour = seed, alpha = 0.1, local_search_optima = 1000, max_attempts = 10, iterations = 5000)

# Plot Solution. Red Point = Initial city; Orange Point = Second City
plot_tour_coordinates(Y, lsgs)
