import random
import matplotlib.pyplot as plt

import networkx as nx
from networkx.algorithms import approximation as approx
from networkx.algorithms.approximation.traveling_salesman import simulated_annealing_tsp

from collections import deque  # Used rotate list
import numpy as np  # Probability of mutation


# Returns fitness value of child (higher is better)
def fitness_function(child_cost):
    fitness = 1 / child_cost
    return fitness


# Genetic Algorithm (OX1) Ordered Crossover with Swap Mutation
# Use for baseline 1 and original chromosomes
def genetic_algorithm_original(parent1, parent2, fitness_function):
    global children
    global baseline_1_tour
    global baseline_1_fitness
    global baseline_1_cost

    # Get 2 random crossover points
    while True:
        pos1 = random.randint(0, chromosome_size)
        pos2 = random.randint(0, chromosome_size)
        if pos1 < pos2:
            break

    # Create the child
    child = []
    for i in range(chromosome_size):
        child.append(-1)

    # Copy segment of parent1 into child
    for i in range(pos1, pos2, 1):
        child[i] = parent1[i]

    # Store elements of parent1 that are not in child yet
    leftover_parent1 = []
    for i in range(chromosome_size):
        if parent1[i] not in child:
            leftover_parent1.append(parent1[i])

    # Organize parent 2 by rotating by elements
    copy = parent2.copy()
    copy = deque(copy)
    copy.rotate(chromosome_size - pos2 - 1)
    copy = list(copy)

    # Order elements
    anti_parent1 = []
    for i in range(chromosome_size):
        if copy[i] in leftover_parent1:
            anti_parent1.append(copy[i])

    # Merge lists to complete the child
    for i in range(len(anti_parent1)):
        index = (pos2 + i) % chromosome_size
        child[index] = anti_parent1[i]

    # Mutate
    if np.random.uniform(0, 1) <= .1:
        while True:
            pos1 = random.randint(0, chromosome_size - 1)
            pos2 = random.randint(0, chromosome_size - 1)
            if pos1 != pos2:
                break

        child[pos1], child[pos2] = child[pos2], child[pos1]

    # Determine fitness value
    child_cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(child))
    child_fitness_value = fitness_function(child_cost)
    if child_fitness_value > baseline_1_fitness:
        baseline_1_fitness = child_fitness_value
        baseline_1_tour = child.copy()

    children.append(child)


# Genetic Algorithm (OX1) Ordered Crossover with Swap Mutation
# Use for alpha chromosomes
def genetic_algorithm_alpha(parent1, parent2, fitness_function):
    global alpha_children
    global alpha_tour
    global alpha_fitness
    global alpha_cost

    # get 2 random crossover points
    while True:
        pos1 = random.randint(0, chromosome_size)
        pos2 = random.randint(0, chromosome_size)
        if pos1 < pos2:
            break

    # create the child
    child = []
    for i in range(chromosome_size):
        child.append(-1)

    # copy segment of parent1 into child
    for i in range(pos1, pos2, 1):
        child[i] = parent1[i]

    # store elements of parent1 that are not in child yet
    leftover_parent1 = []
    for i in range(chromosome_size):
        if parent1[i] not in child:
            leftover_parent1.append(parent1[i])

    # organize parent 2 by rotating by elements
    copy = parent2.copy()
    copy = deque(copy)
    copy.rotate(chromosome_size - pos2 - 1)
    copy = list(copy)

    # order elements
    anti_parent1 = []
    for i in range(chromosome_size):
        if copy[i] in leftover_parent1:
            anti_parent1.append(copy[i])

    # merge lists to complete the child
    for i in range(len(anti_parent1)):
        index = (pos2 + i) % chromosome_size
        child[index] = anti_parent1[i]

    # mutate
    if np.random.uniform(0, 1) <= .1:
        while True:
            pos1 = random.randint(0, chromosome_size - 1)
            pos2 = random.randint(0, chromosome_size - 1)
            if pos1 != pos2:
                break

        child[pos1], child[pos2] = child[pos2], child[pos1]

    # determine fitness value
    child_cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(child))
    child_fitness_value = fitness_function(child_cost)
    if (child_fitness_value > alpha_fitness):
        alpha_fitness = child_fitness_value
        alpha_tour = child.copy()

    alpha_children.append(child)


# ----- GRAPH CREATION AND PLOT ------
# Create graph object
G = nx.complete_graph(50)
pos = nx.spring_layout(G)

# Add random weights
for (u, v, w) in G.edges(data=True):
    w['weight'] = random.randint(10, 200)

# # Plot graph
# plt.figure(figsize=(50,50))
# nx.draw(G, pos, node_size=1200, node_color='lightblue',
#         linewidths=0.25, font_size=10,
#         font_weight='bold', with_labels=True)
#
# # Add weight labels
# labels = nx.get_edge_attributes(G,'weight')
# nx.draw_networkx_edge_labels(G, pos,edge_labels=labels)
#
# plt.show()

# ----- CREATE POPULATION -----
chromosomes = []  # list of list for chromosomes
source = 3
original_chromosome = approx.greedy_tsp(G, source=source)
original_chromosome_cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(original_chromosome))
original_chromosome.pop(0)
original_chromosome.pop(-1)

chromosome_size = len(original_chromosome)

# generate 500 chromosomes
for i in range(500):
    # copy chromosome into A
    A = original_chromosome.copy()

    # swap two random elements
    while True:
        pos1 = random.randint(0, chromosome_size - 1)
        pos2 = random.randint(0, chromosome_size - 1)
        if pos1 != pos2:
            break

    A[pos1], A[pos2] = A[pos2], A[pos1]

    # add new chromosome into list of chromosomes
    chromosomes.append(A)

# ----- FIND BASELINE 1: WITH GENETIC ALGORITHM
children = []
baseline_1_tour = []
baseline_1_fitness = 0
baseline_1_cost = 0

index_parent1 = 0
index_parent2 = 1

while len(children) != 250:
    genetic_algorithm_original(chromosomes[index_parent1], chromosomes[index_parent2], fitness_function)
    genetic_algorithm_original(chromosomes[index_parent2], chromosomes[index_parent1], fitness_function)
    index_parent1 += 2
    index_parent2 += 2

# Add starting node to the beginning and end of baseline 1 to make it a cycle
baseline_1_tour.insert(0, source)
baseline_1_tour.append(source)

baseline_1_cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(baseline_1_tour))

# ----- FIND BASELINE 2: WITH SIMULATED ANNEALING
baseline_2_tour = approx.simulated_annealing_tsp(G, "greedy", source=source)
baseline_2_cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(baseline_2_tour))

# ----- FIND ALPHA BASELINE: WITH SIMULATED ANNEALING + GENETIC ALGORITHM
alpha_chromosomes_pre = []
alpha_chromosomes_after = []
alpha_children = []
alpha_tour = []
alpha_fitness = 0
alpha_cost = 0

# generate alpha values
alpha_values = np.arange(0.001, 1.001, 0.001)

# populate alpha_tours
for i in range(len(alpha_values)):
    alpha_chromosomes_pre.append(approx.simulated_annealing_tsp(G, "greedy", alpha=i, source=source))
    alpha_chromosomes_pre[i].pop(0)
    alpha_chromosomes_pre[i].pop(-1)

# swap
for i in range(len(alpha_chromosomes_pre)):
    # copy chromosome into A
    A = alpha_chromosomes_pre[i].copy()

    # swap two random elements
    while True:
        pos1 = random.randint(0, chromosome_size - 1)
        pos2 = random.randint(0, chromosome_size - 1)
        if pos1 != pos2:
            break

    A[pos1], A[pos2] = A[pos2], A[pos1]

    # add new chromosome into list of chromosomes
    alpha_chromosomes_after.append(A)

# run alpha_tours through GA
index_parent1 = 0
index_parent2 = 1

while len(alpha_children) != 500:
    genetic_algorithm_alpha(alpha_chromosomes_after[index_parent1], alpha_chromosomes_after[index_parent2], fitness_function)
    genetic_algorithm_alpha(alpha_chromosomes_after[index_parent2], alpha_chromosomes_after[index_parent1], fitness_function)
    index_parent1 += 2
    index_parent2 += 2

# add starting node to the beginning and end of baseline 1 to make it a cycle
alpha_tour.insert(0, source)
alpha_tour.append(source)

alpha_cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(alpha_tour))

# ----- PRINT DATA
# - print original chromosome
# print(f"Original Tour: {original_chromosome}")
# print(f"Original Cost: {original_chromosome_cost}")

# - print baseline_1
# print(f"Baseline 1 Tour: {baseline_1_tour}")
print(f"Baseline 1 Cost: {baseline_1_cost}")

# - print baseline_2
# print(f"Baseline 2 Tour: {baseline_2_tour}")
print(f"Baseline 2 Cost: {baseline_2_cost}")

# - print alpha
# print(f"Alpha Tour: {alpha_tour}")
print(f"Alpha Cost: {alpha_cost}")

# for i in range(len(chromosomes)):
# print(f"chromosome {i}:\t {chromosomes[i]}")

# - print children
# for i in range(len(children)):
# print(f"Children {i}:\t {children[i]}")
