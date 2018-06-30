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
import random
import copy

# Function: Distance
def distance_calc(Xdata, route):
    distance = 0
    for k in range(0, len(route[0])-1):
        m = k + 1
        distance = distance + Xdata.iloc[route[0][k]-1, route[0][m]-1]            
    return distance

# Function: Stochastic 2_opt
def stochastic_2_opt(Xdata, city_tour):
    best_route = copy.deepcopy(city_tour)      
    i, j  = random.sample(range(0, len(city_tour[0])-1), 2)
    if (i > j):
        i, j = j, i
    best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))           
    best_route[0][-1]  = best_route[0][0]              
    best_route[1] = distance_calc(Xdata, route = best_route)                     
    return best_route

#Function: Augmented Cost
def augumented_cost(Xdata, city_tour, penalty, limit):
    distance = 0
    augmented = 0   
    for i in range(0, len(city_tour[0]) - 1):
        c1 = city_tour[0][i]
        c2 = city_tour[0][i + 1]      
        if c2 < c1:
            c1, c2 = c2, c1            
        distance  = distance  + Xdata.iloc[c1-1, c2-1]
        augmented = augmented + Xdata.iloc[c1-1, c2-1] + (limit * penalty[c1-1][c2-1])    
    return [distance, augmented]

# Function: Local Search
def local_search(Xdata, city_tour, penalty, max_attempts = 50, limit= 1):
    count = 0
    _, ag_cost = augumented_cost(Xdata, city_tour = city_tour, penalty = penalty, limit = limit)
    solution = copy.deepcopy(city_tour) 
    while (count < max_attempts):
        candidate = stochastic_2_opt(Xdata, city_tour = solution)
        _, candidate_augmented = augumented_cost(Xdata, city_tour = candidate, penalty = penalty, limit = limit)       
        if candidate_augmented < ag_cost:
            solution  = copy.deepcopy(candidate)
            _, ag_cost = augumented_cost(Xdata, city_tour = solution, penalty = penalty, limit = limit)
            count = 0
        else:
            count = count + 1                             
    return solution 

def utility (Xdata, city_tour, penalty, limit = 1):
    utilities = [0 for i in city_tour[0]]
    for i in range(0, len(city_tour[0]) - 1):
        c1 = city_tour[0][i]
        c2 = city_tour[0][i + 1]      
        if c2 < c1:
            c1, c2 = c2, c1            
        utilities[i] = Xdata.iloc[c1-1, c2-1] /(1 + penalty[c1-1][c2-1])  
    return utilities

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
def guided_search(Xdata, city_tour, alpha = 0.3, local_search_optima = 12000, max_attempts = 20, iterations = 50):
    count = 0
    limit = alpha * (local_search_optima / len(city_tour[0]))  
    penalty = [[0 for i in city_tour[0]] for j in city_tour[0]]
    solution = copy.deepcopy(city_tour)
    best_solution = [[],float("inf")]
    while (count < iterations):
        solution = local_search(Xdata, city_tour = solution, penalty = penalty, max_attempts = max_attempts, limit = limit)
        utilities = utility (Xdata, city_tour = solution, penalty = penalty, limit = limit)
        penalty = update_penalty(penalty = penalty, city_tour = solution, utilities = utilities)
        if (solution[1] < best_solution[1]):
            best_solution = copy.deepcopy(solution) 
        count = count + 1
        if (count > 0):
            print("Iteration = ", count, "->", best_solution)
    return best_solution, penalty

######################## Part 1 - Usage ####################################

X = pd.read_csv('Python-MH-Local Search-Guided Search-Dataset-01.txt', sep = '\t') #17 cities = 2085

cities = [[   1,  2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,   1   ], 4722]
lsgs = guided_search(X, city_tour = cities, alpha = 0.3, local_search_optima = 1000, max_attempts = 10, iterations = 1000)