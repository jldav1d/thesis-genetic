
import random
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
import os 
import pandas as pd
import time


# best performing parameters so far
POPULATION_SIZE = 16
MAX_ITERATIONS = 25
CROSSOVER_PROBABILITY = 0.5
ADJUSTMENT_RATIO = 0.05
MUTATION_RATE = 0.4
nnh_ratio = 0.5

TSP_INSTANCE = "kroA100"

directory = f"proposed_algorithm/test_data/{TSP_INSTANCE}/generations"
route_directory = f"proposed_algorithm/test_data/{TSP_INSTANCE}/generated_routes"
excel_directory = f"proposed_algorithm/test_data/{TSP_INSTANCE}/excel"

os.makedirs(directory, exist_ok=True)
os.makedirs(route_directory, exist_ok=True)
os.makedirs(excel_directory, exist_ok=True)

"""
READ:
    Before running the algorithm, Go to Line 55.
"""

def read_tsp_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    node_section = False
    coordinates = {}

    for line in lines:
        if "NODE_COORD_SECTION" in line:
            node_section = True
            continue
        if "EOF" in line:
            break
        if node_section:
            parts = line.strip().split()
            node_id = int(parts[0])
            x, y = float(parts[1]), float(parts[2])
            coordinates[node_id] = (x, y)
    
    return coordinates

def compute_distance_matrix(coords):
    n = len(coords)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                x1, y1 = coords[i+1]
                x2, y2 = coords[j+1]
                distance_matrix[i, j] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return distance_matrix

"""
READ:
    MAKE SURE TO RUN THE INTENDED TSP INSTANCES
"""
coordinates = read_tsp_file(f"../tsp_instances/{TSP_INSTANCE}.tsp")
distance_matrix = compute_distance_matrix(coordinates)
num_nodes = len(coordinates)
MAX_CUT_RANGE = int(num_nodes * 0.2)



# Fitness function for TSP (minimize total distance)
def fitness(solution, distance_matrix):
    total_distance = 0
    num_cities = len(solution)
    
    # Sum the distances between consecutive cities in the solution
    for i in range(1, num_cities):
        total_distance += distance_matrix[solution[i-1] - 1][solution[i] - 1]
    
    # Add the distance from the last city back to the first city
    total_distance += distance_matrix[solution[-1] - 1][solution[0] - 1]
    # No need to add the distance from last city to first as it's already done
    return total_distance


# Helper function to create a random solution (random permutation of cities)
# Step 1: Initialization
def initialize_mixed_population(population_size, distance_matrix, nnh_ratio):
    """
    Initializes a population with a mix of Nearest Neighbor Heuristic and random solutions.

    Parameters:
    - population_size: Total number of solutions in the population.
    - distance_matrix: Matrix containing the distances between all pairs of cities.
    - nnh_ratio: Proportion of the population to generate using NNH (between 0 and 1).

    Returns:
    - A list representing the mixed population.
    """
    # Determine the number of solutions for each method
    nnh_population_size = int(population_size * nnh_ratio)
    random_population_size = population_size - nnh_population_size

    # Initialize part of the population using Nearest Neighbor Heuristic
    nnh_population = initialize_population_with_nearest_neighbor(nnh_population_size, distance_matrix)

    # Initialize the remainder of the population randomly
    random_population = initialize_population(random_population_size)

    # Combine both parts to form the complete population
    population = nnh_population + random_population

    return population

def create_random_solution():
    return random.sample(range(1, num_nodes + 1), num_nodes)

def initialize_population_with_nearest_neighbor(population_size, distance_matrix):
    population = []
    num_cities = len(distance_matrix)

    # Generate a starting tour using nearest neighbor from different initial cities
    for _ in range(population_size):
        start_city = random.randint(1, num_cities)  # Random start city in 1-based index
        tour = nearest_neighbor_solution(start_city, distance_matrix)
        population.append(tour)

    return population

def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        route = create_random_solution()

        # refine route
        route = search_method(route)
        population.append(route)
    return population

# New Version Nearest Neighbor Population Initialization (All)
def nearest_neighbor_solution(start_city, distance_matrix):
    num_cities = len(distance_matrix)
    unvisited = set(range(1, num_cities + 1))
    unvisited.remove(start_city)
    tour = [start_city]

    current_city = start_city
    while unvisited:
        # Find the nearest neighbor; adjust for zero-based indexing if necessary in your dataset
        next_city = min(
            unvisited,
            key=lambda city: distance_matrix[current_city - 1, city - 1]
        )
        tour.append(next_city)
        unvisited.remove(next_city)
        current_city = next_city

    return tour


# Step 2: Evaluation
def evaluate_population(population):
    return [fitness(candidate, distance_matrix) for candidate in population]

def stochastic_universal_sampling(population, adjustment_ratio):
    population_fitness = evaluate_population(population)

    # Adjust fitness based on the adjustment_ratio
    if adjustment_ratio > 0.0:
        # Find the max fitness
        max_fitness = max(population_fitness)

        # Identify the dominant candidates (you can define a threshold, e.g., above median)
        threshold = sorted(population_fitness)[len(population) // 2]
        dominant_candidates = [i for i, fitness in enumerate(population_fitness) if fitness >= threshold]

        # Calculate the total amount of dominance to redistribute
        total_dominance = 0
        for i in dominant_candidates:
            total_dominance += (population_fitness[i] - threshold) * (adjustment_ratio / 100.0)

        # Distribute this dominance equally among weaker candidates
        num_weaker_candidates = len(population) - len(dominant_candidates)
        distribution_amount = total_dominance / num_weaker_candidates if num_weaker_candidates > 0 else 0

        # Update fitness for weaker candidates
        for i in range(len(population_fitness)):
            if i not in dominant_candidates:
                population_fitness[i] += distribution_amount

    total_fitness = sum(population_fitness)
    distance = total_fitness / POPULATION_SIZE
    start_point = random.uniform(0, distance)
    pointers = [start_point + i * distance for i in range(POPULATION_SIZE)]
    selected_population = []

    for pointer in pointers:
        cumulative_fitness = 0
        for i, candidate in enumerate(population):
            cumulative_fitness += population_fitness[i]
            if cumulative_fitness >= pointer:
                selected_population.append(candidate)
                break

    return selected_population

# 2-Opt Swap: Reverses the order of nodes between indices i and k
def two_opt_swap(route, i, k):
    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
    return new_route

# 2-Opt Local Search
def two_opt(route, distance_matrix = distance_matrix, max_iterations = 25):
    best_route = route
    best_distance = fitness(best_route, distance_matrix)
    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        for i in range(1, len(route) - 2):
            for k in range(i + 1, len(route) - 1):
                new_route = two_opt_swap(best_route, i, k)
                new_distance = fitness(new_route, distance_matrix)
                
                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    improved = True  # Continue searching if improvement found
        iteration += 1

    return best_route

def search_method(route):
    size = len(route)
    cut_point = int(size/2)

    add_point = 0 if (size % 2 == 0) else 1  

    best_distance = fitness(route, distance_matrix)

    # add extra element for segment1 if size is odd
    segment1 = list(route[:cut_point])
    segment2 = list(route[cut_point:])

    current_route = []
    for i in range(cut_point):
        idx_i, idx_j = 0,0
        current_distance = 99999999999
        lowest_distance = current_distance
        count = 0
        
        for j in segment2:

            current_distance = (distance_matrix[segment1[i]-1][segment2[count]-1])
                        
            if current_distance < lowest_distance:
                lowest_distance = current_distance
                idx_i = i
                idx_j = count

            count += 1

        current_route.append(segment1[idx_i])
        current_route.append(segment2[idx_j])
        segment2.pop(idx_j)

    if add_point == 1:
        current_route.append(segment2[-1])
    
    before_swap_distance = fitness(current_route, distance_matrix)
   
    # swap first and last element
    current_route[0], current_route[-1] = current_route[-1], current_route[0]

    after_swap_distance = fitness(current_route, distance_matrix)

    if after_swap_distance < before_swap_distance:
        return current_route
    
    # undo swapping
    current_route[0], current_route[-1] = current_route[-1], current_route[0]

    return current_route

def proposed_local_search(route, max_iterations = 2):
    improved = True
    iterations = 0
    best_route = route
    best_distance = fitness(best_route, distance_matrix)

    while improved and iterations < max_iterations:
        improved = False
        new_route = search_method(best_route)
        new_distance = fitness(new_route, distance_matrix)

        if new_distance < best_distance:
            best_distance = new_distance
            best_route = new_route
            improved = True

    return best_route


def balanced_or_opt(route, sample_fraction=0.3):
    """Or-Opt with a more balanced approach: samples 50% of positions for better quality."""
    size = len(route)
    best_route = route[:]
    best_distance = fitness(best_route, distance_matrix)

    segment_sizes = [1, 2, 3]
    num_samples = max(1, int(sample_fraction * size))  # Use 50% sampling

    for segment_size in segment_sizes:
        for _ in range(num_samples):  
            start = random.randint(0, size - segment_size - 1)
            segment = best_route[start:start + segment_size]
            remaining_route = best_route[:start] + best_route[start + segment_size:]

            insert_positions = random.sample(range(len(remaining_route) + 1), min(8, len(remaining_route)))  # Sample 8 positions

            best_local_route = None
            best_local_distance = float('inf')

            for insert_pos in insert_positions:
                if insert_pos == start:
                    continue

                new_route = remaining_route[:insert_pos] + segment + remaining_route[insert_pos:]
                new_distance = fitness(new_route, distance_matrix)

                if new_distance < best_local_distance:
                    best_local_route = new_route
                    best_local_distance = new_distance

            if best_local_route and best_local_distance < best_distance:
                return best_local_route  # Accept best sampled move

    return best_route

def ranked_two_point_swap(route, base_ratio=0.1):
    """Adaptive 2.5-opt swap: dynamically adjusts max_swaps based on route size and edge disparity."""
    size = len(route)
    best_route = route[:]
    best_distance = fitness(best_route, distance_matrix)

    edge_scores = [(i, distance_matrix[route[i] - 1][route[i + 1] - 1]) for i in range(size - 1)]
    edge_scores.sort(key=lambda x: -x[1])  # Sort by worst edges

    # Adaptive max_swaps: proportional to route size and worst-edge variance
    edge_distances = [d for _, d in edge_scores]
    variance = np.var(edge_distances)  
    max_swaps = int(base_ratio * size * (1 + variance / np.mean(edge_distances)))

    for i, _ in edge_scores[:max_swaps]:  
        if i + 2 >= size:
            continue
        j = random.randint(i + 2, size - 1)  

        new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
        new_distance = fitness(new_route, distance_matrix)

        if new_distance < best_distance:
            return new_route  

    return best_route

def improved_local_search_2(route, max_iterations=100, no_improve_limit=5):
    """Local search with Or-Opt and 2.5-Opt, augmented by Tabu Search on stagnation."""
    best_route = route
    best_distance = fitness(best_route, distance_matrix)
    no_improve_count = 0

    for iteration in range(max_iterations):
        new_route = balanced_or_opt(best_route, sample_fraction=0.3)
        new_route = ranked_two_point_swap(new_route)
        new_distance = fitness(new_route, distance_matrix)
        if new_distance < best_distance:
            best_route = new_route
            best_distance = new_distance
            no_improve_count = 0
        else:
            no_improve_count += 1

        # if iteration % 10 == 0:
        #     new_route = two_opt(best_route, distance_matrix, max_iterations=5)

        # If stagnant, perform a Tabu Search step to escape the local optimum
        if no_improve_count >= no_improve_limit:
            new_route = two_opt(best_route, distance_matrix, max_iterations=max_iterations-iteration)
            new_distance = fitness(new_route, distance_matrix)
            if new_distance < best_distance:
                best_route = new_route
                best_distance = new_distance
                no_improve_count = 0
            else:
                break 

    return best_route

def adaptive_order_crossover(parent1, parent2, max_cut_range=MAX_CUT_RANGE, min_cut_range=5):
    """
    Adaptive Order Crossover: selects dynamic cut points based on parent fitness.
    """
    size = len(parent1)
    
    # Compute fitness of parents
    fitness_parent1 = fitness(parent1, distance_matrix)
    fitness_parent2 = fitness(parent2, distance_matrix)
    
    # Use fitness to determine the cut point range
    if fitness_parent1 < fitness_parent2:
        cut_range = max_cut_range
    else:
        cut_range = min_cut_range
    
    # Randomly select cut points within the determined range
    cut1 = random.randint(2, size - cut_range)
    cut2 = random.randint(cut1 + 1, min(size, cut1 + cut_range))
    
    # Ensure cut2 is greater than cut1
    if cut1 > cut2:
        cut1, cut2 = cut2, cut1
    
    # Perform the order crossover
    parent1_copy = parent1[:]
    parent2_copy = parent2[:]
    
    child1, child2 = [None] * size, [None] * size
    
    # Copy substring between cut points
    child1[cut1:cut2] = parent1_copy[cut1:cut2]
    child2[cut1:cut2] = parent2_copy[cut1:cut2]
    
    def fill_child(child, parent):
        pos = cut2  # Start filling from the second cut point
        idx = 0
        while None in child:
            if idx not in range(cut1, cut2):
                child[pos % size] = parent[idx]
                pos += 1
            idx += 1
        return child
    
    # Fill the children
    child1 = fill_child(child1, parent1_copy)
    child2 = fill_child(child2, parent2_copy)
    
    return child1, child2

def scramble_mutation(candidate):
    # Check if the candidate is valid
    if candidate is None:
        return None
    
    # Select two distinct points randomly
    point1, point2 = sorted(random.sample(range(len(candidate)), 2))
    
    # Scramble the genes between the selected points
    genes_to_scramble = candidate[point1:point2]
    random.shuffle(genes_to_scramble)
    candidate[point1:point2] = genes_to_scramble
    
    return candidate

# Termination criterion
def should_terminate(generation):
    return generation >= MAX_ITERATIONS

# Genetic Algorithm
def genetic_algorithm(run = 0):

    filename = f"{directory}/run{run}_{TSP_INSTANCE}_generation.txt"
    start_time = time.time()

    population = initialize_mixed_population(POPULATION_SIZE, distance_matrix, nnh_ratio)
    best_solution = population[0] 
    best_fitness = fitness(best_solution, distance_matrix) 
    best_fitnesses = [best_fitness] 
    average_fitnesses = [sum(evaluate_population(population)) / POPULATION_SIZE] 

    # Create a DataFrame to store per-generation metrics.
    df_metrics = pd.DataFrame(columns=[
        'generation', 'best_fitness', 'average_fitness', 'fitness_std', 'avg_euclidean_distance'
    ])
    
    # Record initial metrics (Generation 0)
    initial_population_fitness = evaluate_population(population)
    initial_avg_fitness = sum(initial_population_fitness) / POPULATION_SIZE
    initial_fitness_std = np.std(initial_population_fitness)
    
    # Compute initial genotypic diversity (average Euclidean distance)
    total_distance = 0.0
    pair_count = 0
    for i in range(len(population)):
        genome_i = np.array(population[i])
        for j in range(i + 1, len(population)):
            genome_j = np.array(population[j])
            total_distance += np.linalg.norm(genome_i - genome_j)
            pair_count += 1
    initial_avg_euclidean_distance = total_distance / pair_count if pair_count > 0 else 0
    
    # Append generation 0 metrics to the DataFrame.
    df_metrics = pd.concat([df_metrics, pd.DataFrame({
        'generation': [0],
        'best_fitness': [best_fitness],
        'average_fitness': [initial_avg_fitness],
        'fitness_std': [initial_fitness_std],
        'avg_euclidean_distance': [initial_avg_euclidean_distance]
    })], ignore_index=True)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for generation in range(MAX_ITERATIONS):
            fitness_values = []
            population_fitness = evaluate_population(population)
            fitness_values.append(population_fitness)

            # increase the mutation rate over time
            # MUTATION_RATE = generation/MAX_ITERATIONS
            
            # Update best_solution and best_fitness
            min_fitness_index = population_fitness.index(min(population_fitness))  # We minimize the fitness (distance)
            if population_fitness[min_fitness_index] < best_fitness:
                best_solution = population[min_fitness_index]
                best_fitness = population_fitness[min_fitness_index]


            # === Diversity Measurements Start ===
            # 1. Fitness Diversity: Standard deviation of fitness values.
            fitness_std = np.std(population_fitness)
            
            # 2. Genotypic Diversity: Average Euclidean distance.
            total_distance = 0.0
            pair_count = 0
            for i in range(len(population)):
                genome_i = np.array(population[i])
                for j in range(i + 1, len(population)):
                    genome_j = np.array(population[j])
                    total_distance += np.linalg.norm(genome_i - genome_j)
                    pair_count += 1
            avg_euclidean_distance = total_distance / pair_count if pair_count > 0 else 0
            # === Diversity Measurements End ===

            # Step 3: Selection (Stochastic Universal Sampling)
            selected_population = stochastic_universal_sampling(population,ADJUSTMENT_RATIO)

            # Step 4: Crossover (Uniform Crossover)
            offspring = []
            for i in range(0, POPULATION_SIZE, 2):
                parent1 = selected_population[i]
                parent2 = selected_population[i + 1]

                if random.random() >= CROSSOVER_PROBABILITY:
                    child1, child2 = adaptive_order_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2

                offspring.append(child1)
                offspring.append(child2)

            # Step 5: Mutation (Scramble Mutation)
            for i in range(len(offspring)):
                if random.random() < MUTATION_RATE:  
                    offspring[i] = scramble_mutation(offspring[i])

            # run proposed local search
            offspring = list(executor.map(improved_local_search_2, offspring))
      

            # Replace the old population with the new offspring population
            population = offspring

            # Print generation statistics
            average_fitness = sum(population_fitness) / POPULATION_SIZE
            # Print generation statistics.
            print(f"Generation {generation}: Best Fitness = {best_fitness}, "
                  f"Average Fitness = {average_fitness}, Fitness STD = {fitness_std:.2f}, "
                  f"Avg Euclidean Distance = {avg_euclidean_distance:.2f}")
            
            # Append the metrics for this generation to the DataFrame.
            new_row = pd.DataFrame({
                'generation': [generation],
                'best_fitness': [best_fitness],
                'average_fitness': [average_fitness],
                'fitness_std': [fitness_std],
                'avg_euclidean_distance': [avg_euclidean_distance]
            })
            df_metrics = pd.concat([df_metrics, new_row], ignore_index=True)
            
            # Check termination criterion
            if should_terminate(generation):
                break

            # Append best_fitness to the list
            best_fitnesses.append(best_fitness)
            average_fitnesses.append(average_fitness)

    plot_best_route(best_solution, coordinates, run) 
    

    # Final display of the best solution
    print("\nBest Solution (Tour):", best_solution)
    print("Total Distance of the Best Solution:", best_fitness)

    # ----- Export Metrics as Files -----
    # Save as an Excel file.
    df_metrics.to_excel(f"{excel_directory}/run{run}_{TSP_INSTANCE}_metrics_results.xlsx", index=False)
    
    end_time = time.time()
    running_time = end_time - start_time
    print(f"Algorithm ran in {running_time:.2f} seconds")

    initial_best_fitness = min(initial_population_fitness)


    # Optionally, also save as a text file.
    with open(filename, "w") as f:
        f.write(f"TSP Instance: {TSP_INSTANCE}\n")
        f.write(f"\nTotal Running Time: {running_time:.2f} seconds\n")
        f.write("Initial Population's Best Fitness: {:.2f}\n".format(initial_best_fitness))
        f.write("Initial Population's Average Fitness: {:.2f}\n".format(initial_avg_fitness))
        for _, row in df_metrics.iterrows():
            f.write("Generation {}: Best Fitness = {:.2f}, Average Fitness = {:.2f}, "
                    "Fitness STD = {:.2f}, Avg Euclidean Distance = {:.2f}\n".format(
                        int(row['generation']),
                        row['best_fitness'],
                        row['average_fitness'],
                        row['fitness_std'],
                        row['avg_euclidean_distance']
                    ))
        # include best route
        f.write("\nBest Route: {}\n".format(best_solution)) 
        f.write("Best Fitness: {:.2f}\n".format(best_fitness))


def plot_best_route(tour, coordinates, run):
    # Ensure indices in tour map properly
    x = []
    y = []
    # Correct any inadvertent zero indices, real city indices should be between 1 and num_nodes
    for city in tour:
        if city == 0:
            continue  # skip or modify to a valid node if necessary
        x.append(coordinates[city][0])
        y.append(coordinates[city][1])

    # Close the loop by returning to the start city
    x.append(coordinates[tour[0]][0])
    y.append(coordinates[tour[0]][1])

    plt.figure(figsize=(10, 6), dpi=150)  # Higher DPI for clarity
    plt.plot(x, y, marker='o', linestyle='-', markersize=6, linewidth=1.5, color='b', alpha=0.8)


    plt.title('Best Route for TSP')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    for i, city in enumerate(tour):
        if city == 0:
            continue
        plt.annotate(
            f"{city}", 
            (coordinates[city][0], coordinates[city][1]),  
            textcoords="offset points",  
            xytext=(0, 8), 
            ha='center', 
            fontsize=5,  
            fontweight='bold', 
            color='black',  
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="white", facecolor="lightgray", alpha=0.7)
        )
    plt.annotate(
        "Start & End",
        (coordinates[tour[0]][0], coordinates[tour[0]][1]),
        textcoords="offset points",
        xytext=(0, 0),  # Move it a bit above
        ha='center',
        fontsize=5,
        fontweight='bold',
        color='red',  # Make it stand out
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="red", facecolor="white", alpha=0.8)
    )


    plt.grid(True)
    plt.savefig(f"{route_directory}/run{run}_{TSP_INSTANCE}_best_route.png")

# Main entry point
if __name__ == "__main__":
    MAX_RUNS = 25
    for run in range(1, MAX_RUNS):
        genetic_algorithm(run=run + 1)