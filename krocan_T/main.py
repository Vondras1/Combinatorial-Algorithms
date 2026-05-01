#!/usr/bin/env python3
import numpy as np
import sys
import random
from dataclasses import dataclass
import time


class Input:
    def __init__(self):
        self.N = None       # number of customers
        self.K = None       # maximum number of vans (valid only for optimal solutions, heuristic solutions may use more vans)
        self.Q = None       # capacity of the van
        self.Gamma = None   # cost of using a van
        self.s = []         # size per customer 1..N
        self.T_in = []      # earliest time window
        self.T_out = []     # latest time window
        self.T = []         # (N+1)x(N+1) travel times, node 0 is depot
        self.C = []         # (N+1)x(N+1) travel costs, node 0 is depot
    
    def decode_input(self, file):
        input_lines = self.load_input(file)
        self.decode_first_line(input_lines)
        start = 1
        end = start + self.N
        self.decode_customer_specific_info(input_lines, start, end)
        start = end
        end = start + self.N + 1
        self.decode_T(input_lines, start, end)
        start = end
        end = start + self.N + 1
        self.decode_C(input_lines, start, end)

    def load_input(self, file_path = "input.txt"):
        # Open the file in read mode
        file = open(file_path, "r")
        # Read input
        input_lines = []
        for line in file:
            input_lines.append([int(char) for char in line.strip().split()]) # .strip() to remove newline characters
        file.close()
        return input_lines
    
    def decode_first_line(self, input_lines):
        assert len(input_lines[0]) == 4, f"Inappropriate shape of the input (It is {len(input_lines[0])} instead OF 4)"
        self.N = input_lines[0][0]
        self.K = input_lines[0][1]
        self.Q = input_lines[0][2]
        self.Gamma = input_lines[0][3]

    def decode_customer_specific_info(self, input_lines, start, end):
        relevant = np.asarray(input_lines[start:end])
        
        # Make customer data 1-indexed (index 0 reserved for depot)
        self.s = np.concatenate(([0], relevant[:, 0]))
        self.T_in = np.concatenate(([0], relevant[:, 1]))
        self.T_out = np.concatenate(([0], relevant[:, 2]))

    def decode_T(self, input_lines, start, end):
        self.T = np.asarray(input_lines[start:end])

    def decode_C(self, input_lines, start, end):
        self.C = np.asarray(input_lines[start:end])
    
    def print_out_loaded_input(self):
        print("----- Loaded input -----")
        print(f"N = {self.N}, K = {self.K}, Q = {self.Q}, Gamma = {self.Gamma}")
        print("-----")
        print(f"s = {self.s}, T_in = {self.T_in}, T_out = {self.T_out}")
        print("-----")
        print(f"T = {self.T}")
        print("-----")
        print(f"C = {self.C}")
        print("------------------------")


# ------------------------------------------------------------------
# Create a representation of the found solution and check its feasibility and objective value. 
# ------------------------------------------------------------------
Route = list[int]
Solution = list[Route]

@dataclass
class RouteEvaluation:
    route: Route                # list of customers in the order they are visited (without depot)
    load: int                   # total load of the van on this route
    arrival_times: list[int]    # arrival time at each customer (in the same order as in route)
    travel_cost: int            # total travel cost of this route
    feasible: bool              # whether the route is feasible (respects time windows and capacity constraints)

@dataclass
class SolutionEvaluation:
    routes: list[RouteEvaluation]   # list of evaluated routes (one per used van)
    total_travel_cost: int          # total travel cost of the solution (sum of travel costs of all routes)
    used_vans: int                  # number of vans used in the solution
    objective: int                  # objective value of the solution (total travel cost + Gamma * used_vans)
    feasible: bool                  # whether the solution is feasible (all routes are feasible)

def evaluate_route(route: Route, inp: Input, stop_on_infeasible=False) -> RouteEvaluation:
    current_node = 0
    current_time = 0
    load = 0
    travel_cost = 0
    arrival_times = []

    feasible = True

    for customer in route:
        load += inp.s[customer]
        if load > inp.Q:
            feasible = False
            if stop_on_infeasible:
                break

        travel_time = inp.T[current_node, customer]
        travel_cost += inp.C[current_node, customer]

        arrival_time = current_time + travel_time

        if arrival_time > inp.T_out[customer]:
            feasible = False
            if stop_on_infeasible:
                break

        # Waiting is allowed
        service_time = max(arrival_time, inp.T_in[customer])

        arrival_times.append(service_time)

        current_time = service_time
        current_node = customer

    travel_cost += inp.C[current_node, 0]

    return RouteEvaluation(
        route=route.copy(),
        load=load,
        arrival_times=arrival_times,
        travel_cost=travel_cost,
        feasible=feasible,
    )

def evaluate_solution(solution: Solution, inp: Input, stop_on_infeasible=False) -> SolutionEvaluation:
    route_evaluations = []
    total_travel_cost = 0
    feasible = True

    # served_customers = []

    for route in solution:
        route_eval = evaluate_route(route, inp, stop_on_infeasible)
        route_evaluations.append(route_eval)

        total_travel_cost += route_eval.travel_cost
        if not route_eval.feasible:
            feasible = False
            if stop_on_infeasible:
                break

        # served_customers.extend(route)

    used_vans = len(solution)
    objective = total_travel_cost + inp.Gamma * used_vans

    # # Each customer must be served exactly once
    # expected_customers = list(range(1, inp.N + 1))
    # if sorted(served_customers) != expected_customers:
    #     feasible = False

    return SolutionEvaluation(
        routes=route_evaluations,
        total_travel_cost=total_travel_cost,
        used_vans=used_vans,
        objective=objective,
        feasible=feasible,
    )


# ------------------------------------------------------------------
# Try to solve it
# ------------------------------------------------------------------
def copy_solution(solution: Solution) -> Solution:
    return [route.copy() for route in solution]

def filter_solution(solution: Solution) -> Solution:
    return [route for route in solution if len(route) > 0]

# Take one customer from one route and insert that customer into another route at some random position.
def generate_relocate_neighbors(solution: Solution, rng: random.Random):
    # Create randomized list of source routes
    from_route_indices = list(range(len(solution)))
    rng.shuffle(from_route_indices)

    # Select the road from which customer will be removed in random order
    for from_route_idx in from_route_indices:

        # Create randomized list of customers in the selected route
        customer_indices = list(range(len(solution[from_route_idx])))
        rng.shuffle(customer_indices)

        # Create randomized list of target routes
        to_route_indices = list(range(len(solution)))
        rng.shuffle(to_route_indices)

        for customer_idx in customer_indices:
            for to_route_idx in to_route_indices:

                # Skip moves within the same route
                if from_route_idx == to_route_idx:
                    continue

                insert_positions = list(range(len(solution[to_route_idx]) + 1))
                rng.shuffle(insert_positions)

                for insert_pos in insert_positions:
                    new_solution = copy_solution(solution)

                    customer = new_solution[from_route_idx].pop(customer_idx)
                    new_solution[to_route_idx].insert(insert_pos, customer)

                    yield filter_solution(new_solution)

# Take one customer from one route and exchange it with one customer from another route.
def generate_swap_neighbors(solution: Solution, rng: random.Random):
    route1_indices = list(range(len(solution)))
    rng.shuffle(route1_indices)

    for route1_idx in route1_indices:
        route2_indices = list(range(route1_idx + 1, len(solution)))
        rng.shuffle(route2_indices)

        idx1_candidates = list(range(len(solution[route1_idx])))
        rng.shuffle(idx1_candidates)

        for route2_idx in route2_indices:
            idx2_candidates = list(range(len(solution[route2_idx])))
            rng.shuffle(idx2_candidates)

            for idx1 in idx1_candidates:
                for idx2 in idx2_candidates:
                    new_solution = copy_solution(solution)

                    new_solution[route1_idx][idx1], new_solution[route2_idx][idx2] = \
                        new_solution[route2_idx][idx2], new_solution[route1_idx][idx1]

                    yield filter_solution(new_solution)

def generate_tail_swap_neighbors(solution: Solution, rng: random.Random):
    route1_indices = list(range(len(solution)))
    rng.shuffle(route1_indices)

    for route1_idx in route1_indices:
        route2_indices = list(range(route1_idx + 1, len(solution)))
        rng.shuffle(route2_indices)

        for route2_idx in route2_indices:
            route1 = solution[route1_idx]
            route2 = solution[route2_idx]

            # Need at least one customer on both sides of the cut
            if len(route1) < 2 or len(route2) < 2:
                continue

            cut1_candidates = list(range(1, len(route1)))
            cut2_candidates = list(range(1, len(route2)))
            rng.shuffle(cut1_candidates)
            rng.shuffle(cut2_candidates)


            for cut1 in cut1_candidates:
                for cut2 in cut2_candidates:
                    new_solution = copy_solution(solution)

                    prefix1 = new_solution[route1_idx][:cut1]
                    suffix1 = new_solution[route1_idx][cut1:]

                    prefix2 = new_solution[route2_idx][:cut2]
                    suffix2 = new_solution[route2_idx][cut2:]

                    new_solution[route1_idx] = prefix1 + suffix2
                    new_solution[route2_idx] = prefix2 + suffix1

                    yield filter_solution(new_solution)


def generate_all_neighbors(solution: Solution, rng: random.Random):
    if rng.random() < 0.5:
        print(":-}{")
        yield from generate_relocate_neighbors(solution, rng)
        yield from generate_swap_neighbors(solution, rng)
    else:
        print(":-}")
        yield from generate_swap_neighbors(solution, rng)
        yield from generate_relocate_neighbors(solution, rng)
    print("DONE")
    # yield from generate_tail_swap_neighbors(solution, rng)


# def find_best_neighbor(solution: Solution, inp: Input):
#     best_solution = filter_solution(copy_solution(solution))
#     best_eval = evaluate_solution(best_solution, inp)

#     for neighbor in generate_all_neighbors(best_solution):
#         neighbor_eval = evaluate_solution(neighbor, inp)

#         if not neighbor_eval.feasible:
#             continue

#         if (not best_eval.feasible) or (neighbor_eval.objective < best_eval.objective):
#             best_solution = neighbor
#             best_eval = neighbor_eval

#     return best_solution, best_eval

def find_first_improving_neighbor(solution: Solution, inp: Input, rng: random.Random, end_time: float):
    current_solution = filter_solution(copy_solution(solution))
    current_eval = evaluate_solution(current_solution, inp)

    for neighbor in generate_all_neighbors(current_solution, rng):

        if time.time() >= end_time:
            break

        # FIXME - It is not neccessary to evaluate whole solution everytime
        # as relocate and swap change only 2 routes, we can keep track of which routes are changed and re-evaluate only those
        neighbor_eval = evaluate_solution(neighbor, inp, stop_on_infeasible=True) 

        if time.time() >= end_time:
            break
        
        if not neighbor_eval.feasible:
            continue

        if neighbor_eval.objective < current_eval.objective:
            return neighbor, neighbor_eval
        

    return current_solution, current_eval

def hill_climbing(solution: Solution, inp: Input, rng: random.Random, end_time: float):
    current_solution = filter_solution(copy_solution(solution))
    current_eval = evaluate_solution(current_solution, inp)

    while True:
        if time.time() >= end_time:
            break

        next_solution, next_eval = find_first_improving_neighbor(current_solution, inp, rng, end_time)

        if next_eval.objective < current_eval.objective:
            current_solution = next_solution
            current_eval = next_eval
        else:
            break

    return current_solution, current_eval

def generate_random_start_solution(inp: Input, rng: random.Random) -> Solution:
    customers = list(range(1, inp.N + 1))
    rng.shuffle(customers)
    return [[customer] for customer in customers]

def restart_hill_climbing(solution: Solution, inp: Input, rng: random.Random, end_time: float):
    best_solution = filter_solution(copy_solution(solution))
    best_eval = evaluate_solution(best_solution, inp)

    while True:
        if time.time() >= end_time:
            break

        start_solution = generate_random_start_solution(inp, rng)
        current_solution, current_eval = hill_climbing(start_solution, inp, rng, end_time)

        if current_eval.objective < best_eval.objective:
            best_solution = current_solution
            best_eval = current_eval

    return best_solution, best_eval

# ------------------------------------------------------------------
# Write the solution to the output file in the required format.
# ------------------------------------------------------------------
def write_solution(filename: str, solution: Solution, inp: Input) -> None:
    # Keep the invariant that empty routes are not written.
    non_empty_solution = [route for route in solution if len(route) > 0]

    solution_eval = evaluate_solution(non_empty_solution, inp, stop_on_infeasible=False)

    with open(filename, "w", encoding="utf-8") as f:
        if not solution_eval.feasible:
            f.write("-1\n")
            return

        # The assignment example prints Obj as a float-like value, e.g. 38.0
        f.write(f"{float(solution_eval.objective)} {solution_eval.used_vans}\n")

        for route_eval in solution_eval.routes:
            route = route_eval.route
            times = route_eval.arrival_times  # currently these are effective visit times after waiting

            f.write(str(len(route)))
            for customer, arrival_time in zip(route, times):
                f.write(f" {customer} {arrival_time}")
            f.write("\n")


def main():
    if len(sys.argv) != 4:
        print("Usage: ./your-solver PATH_INPUT_FILE PATH_OUTPUT_FILE TIME_LIMIT_SECONDS")
        sys.exit(1)
    
    path_input_file = sys.argv[1]
    path_output_file = sys.argv[2]
    time_limit = float(sys.argv[3])

    start_time = time.time()
    end_time = start_time + time_limit
    rng = random.Random(42)

    print("Input file:", path_input_file)
    print("Output file:", path_output_file)
    print("Time limit:", time_limit)

    # Load input data
    inp = Input()
    inp.decode_input(path_input_file)

    # # Temporary test solution: each customer in its own route
    # solution = [[customer] for customer in range(1, inp.N + 1)]

    # solution = [route for route in solution if len(route) > 0] # Remove empty routes (if any)
    # solution_eval = evaluate_solution(solution, inp)

    solution = [[customer] for customer in range(1, inp.N + 1)]

    time_reserved_for_writing = 0.05 # seconds # TODO
    solution, solution_eval = restart_hill_climbing(solution, inp, rng, end_time-time_reserved_for_writing) # Leave some time for writing the solution


    print("Feasible:", solution_eval.feasible)
    print("Used vans:", solution_eval.used_vans)
    print("Travel cost:", solution_eval.total_travel_cost)
    print("Objective:", solution_eval.objective)

    write_solution(path_output_file, solution, inp)
    print(f"Endig time: {time.time()}, total time: {time.time() - start_time}")

if __name__ == "__main__":
    main()