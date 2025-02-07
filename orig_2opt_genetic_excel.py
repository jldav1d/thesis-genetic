import random
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures 
import pandas as pd
import time
import os


"""
    WITH EXCEL OUTPUT
"""

# best performing parameters so far
POPULATION_SIZE = 16
MAX_ITERATIONS = 25
CROSSOVER_PROBABILITY = 0.9
MUTATION_RATE = 0.2
TSP_INSTANCE = "kroA100"

directory = f"genetic_plus_2opt/test_data/{TSP_INSTANCE}/generations"
route_directory = f"genetic_plus_2opt/test_data/{TSP_INSTANCE}/generated_routes"
excel_directory = f"genetic_plus_2opt/test_data/{TSP_INSTANCE}/excel"

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


def fitness(solution, distance_matrix):
    total_distance = 0
    num_cities = len(solution)
    
    # Sum the distances between consecutive cities in the solution
    for i in range(1, num_cities):
        total_distance += distance_matrix[solution[i-1] - 1][solution[i] - 1]
    
    # Add the distance from the last city back to the first city
    total_distance += distance_matrix[solution[-1] - 1][solution[0] - 1]
    return total_distance

def create_random_solution():
    return random.sample(range(1, num_nodes + 1), num_nodes)

def initialize_population():
    return [create_random_solution() for _ in range(POPULATION_SIZE)]

def evaluate_population(population):
    return [fitness(candidate, distance_matrix) for candidate in population]

def stochastic_universal_sampling(population):
    population_fitness = evaluate_population(population)
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
def two_opt(route, distance_matrix = distance_matrix, max_iterations = 50):
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


def pmx_crossover(parent1, parent2):
    size = len(parent1)
    child1, child2 = [-1] * size, [-1] * size
    
    # Select two random crossover points
    cx1, cx2 = sorted(random.sample(range(size), 2))
    
    # Copy the mapping section from parents to children
    child1[cx1:cx2+1] = parent1[cx1:cx2+1]
    child2[cx1:cx2+1] = parent2[cx1:cx2+1]
    
    # Helper function to map values
    def map_value(child, parent, value):
        while value in child[cx1:cx2+1]:
            value = parent[child.index(value)]
        return value
    
    # Fill the remaining positions in children
    for i in range(size):
        if i < cx1 or i > cx2:
            child1[i] = map_value(child1, parent2, parent2[i])
            child2[i] = map_value(child2, parent1, parent1[i])
    
    return child1, child2

# Termination criterion
def should_terminate(generation):
    return generation >= MAX_ITERATIONS

# Genetic Algorithm With Excel Output
def genetic_algorithm(run = 0):
    

    filename = f"{directory}/run{run}_{TSP_INSTANCE}_generation.txt"
    start_time = time.time()

    # Initialize the population.
    population = initialize_population()
    
    # Set initial best solution and fitness.
    best_solution = population[0]
    best_fitness = fitness(best_solution, distance_matrix)
    
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
        for generation in range(1, MAX_ITERATIONS):
            # Evaluate the population.
            population_fitness = evaluate_population(population)
            
            # Update best_solution and best_fitness (minimization problem).
            min_fitness_index = population_fitness.index(min(population_fitness))
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
            selected_population = stochastic_universal_sampling(population)
            
            # Step 4: Crossover (Uniform Crossover)
            offspring = []
            for i in range(0, POPULATION_SIZE, 2):
                parent1 = selected_population[i]
                parent2 = selected_population[i + 1]
                if random.random() >= CROSSOVER_PROBABILITY:
                    child1, child2 = pmx_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                offspring.append(child1)
                offspring.append(child2)
            
            # Step 5: Mutation (Scramble Mutation)
            for i in range(len(offspring)):
                if random.random() < MUTATION_RATE:
                    offspring[i] = scramble_mutation(offspring[i])
            
            # Run proposed 2-opt local search in parallel.
            offspring = list(executor.map(two_opt, offspring))
            
            # Replace the population.
            population = offspring
            
            # Compute average fitness.
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
            
            # Check termination criterion.
            if should_terminate(generation):
                break
    
    # Plot the best route.
    plot_best_route(best_solution, coordinates, run)
    
    # Final display.
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
    # plt.show()

# Main entry point
if __name__ == "__main__":
    MAX_RUNS = 25
    for run in range(0, MAX_RUNS):
        genetic_algorithm(run=run + 1)