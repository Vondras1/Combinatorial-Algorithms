#!/usr/bin/env python3
import numpy as np
import gurobipy as g
import sys

class Input:
    def __init__(self):
        self.n = None       # number of stripes
        self.w = None       # width of the stripe
        self.h = None       # height of the stripe
        self.stripes = []
    
    def decode_input(self, file):
        input_lines = self.load_input(file)
        self.decode_first_line(input_lines)
        self.decode_stripes(input_lines)

    def load_input(self, file = "input.txt"):
        # Open the file in read mode
        file = open(file, "r")
        # Read input
        input_lines = []
        for line in file:
            input_lines.append([int(char) for char in line.strip().split()]) # .strip() to remove newline characters
        file.close()
        return input_lines
    
    def decode_first_line(self, input_lines):
        # assert len(input_lines[0]) == 4, f"Inappropriate shape of the input (It is {len(input_lines[0])} instead OF 4)"
        self.n = input_lines[0][0]
        self.w = input_lines[0][1]
        self.h = input_lines[0][2]

    def decode_stripes(self, input_lines):
        # Flatten all except first row
        data = []
        for line in input_lines[1:]:
            data.extend(line)

        data = np.array(data, dtype=int)

        # reshape into (n, h, w, 3)
        self.stripes = data.reshape(self.n, self.h, self.w, 3)
    
    def print_out_loaded_input(self):
        print("----- Loaded input -----")
        print(f"n = {self.n}, w = {self.w}, h = {self.h}")
        print("-----")
        print(f"Stripes = {self.stripes}")
        print("-----")

def extract_cycles_from_successor(successor):
    """
    Given a successor array/list where successor[i] = j,
    return list of all directed cycles.
    Because of TSP degree constraints, every node belongs to exactly one cycle.
    """
    N = len(successor)
    visited = [False] * N
    cycles = []

    for start in range(N):
        if visited[start]:
            continue

        current = start
        path_index = {}
        path = []

        while current not in path_index and not visited[current]:
            path_index[current] = len(path)
            path.append(current)
            current = successor[current]

        if current in path_index:
            cycle = path[path_index[current]:]
            cycles.append(cycle)

        for node in path:
            visited[node] = True

    return cycles


def find_subtour_from_solution(model):
    """
    Read current solution from callback/model and return the shortest subtour.
    """
    x = model._x
    N = model._N

    successor = [-1] * N
    for i in range(N):
        for j in range(N):
            if i != j and model.cbGetSolution(x[i, j]) > 0.5:
                successor[i] = j
                break

    cycles = extract_cycles_from_successor(successor)
    shortest_cycle = min(cycles, key=len)
    return shortest_cycle


def my_callback(model, where):
    """
    Add lazy constraint forbidding the shortest subtour whenever
    an integer feasible solution is found.
    """
    if where == g.GRB.Callback.MIPSOL:
        cycle = find_subtour_from_solution(model)

        # Full tour must contain all nodes, including dummy.
        if len(cycle) < model._N:
            x = model._x
            model.cbLazy(
                g.quicksum(
                    x[i, j]
                    for i in cycle
                    for j in cycle
                    if i != j
                ) <= len(cycle) - 1
            )

# Define optimization problem
def optimization_problem(D_tsp, path_output_file):
    N = D_tsp.shape[0]

    model = g.Model()
    model.Params.LazyConstraints = 1

    x = model.addVars(N,N, vtype=g.GRB.BINARY,name="x")
    # No self-loops
    for i in range(N):
        x[i, i].ub = 0

    # Constrains
    # Exactly one outgoing edge from each node
    model.addConstrs(
        (g.quicksum(x[i, j] for j in range(N)) == 1 for i in range(N)),
        name="out_degree"
    )

    # Exactly one incoming edge to each node
    model.addConstrs(
        (g.quicksum(x[i, j] for i in range(N)) == 1 for j in range(N)),
        name="in_degree"
    )

    # Objective
    model.setObjective(
        g.quicksum(D_tsp[i, j] * x[i, j] for i in range(N) for j in range(N)),
        g.GRB.MINIMIZE
    )

    # Store for callback
    model._x = x
    model._N = N

    model.optimize(my_callback)

    if model.Status != g.GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi did not find an optimal solution. Status = {model.Status}")

    # Reconstruct final tour
    successor = [-1] * N
    for i in range(N):
        for j in range(N):
            if i != j and x[i, j].X > 0.5:
                successor[i] = j
                break

    # Build one cycle
    tour = []
    current = 0
    while True:
        current = successor[current]
        if tour and current == tour[0]:
            break
        # Do not add dummy
        if current != 0:
            tour.append(current)

    with open(path_output_file, "w") as f:
        for stripe in tour:
            f.write(f"{stripe} ")
        f.write("\n")

def main():
    if len(sys.argv) != 3:
        print("Usage: ./your-solver PATH_INPUT_FILE PATH_OUTPUT_FILE")
        sys.exit(1)
    
    path_input_file = sys.argv[1]
    path_output_file = sys.argv[2]

    print("Input file:", path_input_file)
    print("Output file:", path_output_file)

    input = Input()
    input.decode_input(path_input_file)

    # Prepare left and right columns
    right = input.stripes[:,:,-1,:]
    left = input.stripes[:,:,0,:]

    # Using broadcasting compute distance matrix
    D_raw = np.abs(right[:,None,:,:] - left[None,:,:,:]) # For concrete item D_raw[i,j,r,c] = right[i,r,c] - left[j,r,c]
    D = D_raw.sum(axis=(2,3))

    # We do not want a cycle over just the stripes, because the image has a left end and a right end. 
    # -> I will convert shortest Hamiltonian path to TSP by adding one extra (dummy) node
    # D_tsp = np.zeros((input.n+1, input.n+1))
    # D_tsp[:input.n, :input.n] = D
    D_tsp = np.zeros((input.n + 1, input.n + 1), dtype=int)
    D_tsp[1:, 1:] = D

    optimization_problem(D_tsp, path_output_file)


if __name__ == "__main__":
    main()