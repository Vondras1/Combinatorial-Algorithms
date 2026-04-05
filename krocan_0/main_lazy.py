#!/usr/bin/env python3
import numpy as np
import gurobipy as g  # import Gurobi
import sys

class Input:
    def __init__(self):
        self.N = None       # number of customers
        self.K = None       # maximum number of vans
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
        assert len(input_lines[0]) == 4, f"Inappropriate shape of the input (It is {len(input_lines[0])} instead OF 4)"
        self.N = input_lines[0][0]
        self.K = input_lines[0][1]
        self.Q = input_lines[0][2]
        self.Gamma = input_lines[0][3]

    def decode_customer_specific_info(self, input_lines, start, end):
        relevant = np.asarray(input_lines[start:end])
        # self.s = relevant[:, 0]
        # self.T_in = relevant[:, 1]
        # self.T_out = relevant[:, 2]
        
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

def write_solution(filename, m, x, t, z, vehicles, nodes, customers):
    if m.status != g.GRB.OPTIMAL:
        with open(filename, "w") as f:
            f.write("-1\n")
        return

    used_vehicles = [d for d in vehicles if z[d].X > 0.5]

    with open(filename, "w") as f:
        # první řádek
        f.write(f"{m.objVal} {len(used_vehicles)}\n")

        # pro každé použité vozidlo rekonstruuj trasu
        for d in used_vehicles:
            route = []
            current = 0  # start at depot

            while True:
                next_node = None
                for v in nodes:
                    if v != current and x[d, current, v].X > 0.5:
                        next_node = v
                        break

                if next_node is None or next_node == 0:
                    break

                route.append(next_node)
                current = next_node

            # výpis jedné trasy
            f.write(str(len(route)))
            for customer in route:
                arrival_time = int(round(t[d, customer].X))
                f.write(f" {customer} {arrival_time}")
            f.write("\n")

def extract_graph_components(selected_arcs, customers):
    """
    selected_arcs: list of tuples (u, v) for one vehicle with x[d,u,v] = 1
    Returns connected components among customer nodes only.
    Depot 0 is ignored.
    """
    adjacency = {i: set() for i in customers}

    for u, v in selected_arcs:
        if u != 0 and v != 0:
            adjacency[u].add(v)
            adjacency[v].add(u)

    visited = set()
    components = []

    for start in customers:
        # if already visited, skip
        if start in visited:
            continue

        # if it has no customer neighbors at all, skip
        if len(adjacency[start]) == 0:
            continue

        stack = [start]
        component = []
        visited.add(start)

        while stack:
            node = stack.pop()
            component.append(node)
            for nei in adjacency[node]:
                if nei not in visited:
                    visited.add(nei)
                    stack.append(nei)

        components.append(component)

    return components

def subtour_elimination_callback(model, where):
    if where != g.GRB.Callback.MIPSOL:
        return

    x = model._x
    z = model._z
    vehicles = model._vehicles
    nodes = model._nodes
    customers = model._customers

    for d in vehicles:
        # skip unused vehicles
        if model.cbGetSolution(z[d]) < 0.5:
            continue

        # collect selected arcs
        selected_arcs = []
        for u in nodes:
            for v in nodes:
                if u != v and model.cbGetSolution(x[d, u, v]) > 0.5:
                    selected_arcs.append((u, v))


        components = extract_graph_components(selected_arcs, customers)

        for comp in components:
            # Check whether this component is disconnected from depot.
            # If no arc connects depot <=> comp, then it is a forbidden subtour.
            uses_depot = False
            comp_set = set(comp)

            for (u, v) in selected_arcs:
                if (u == 0 and v in comp_set) or (v == 0 and u in comp_set):
                    uses_depot = True
                    break

            if uses_depot:
                continue

            # Add lazy constraint: a subtour needs |S| internal arcs, but the constraint allows at most |S-1|, so the subtour cannot exist.
            # sum_{u,v in comp, u!=v} x[d,u,v] <= |comp| - 1
            model.cbLazy(g.quicksum(x[d, u, v] for u in comp for v in comp if u != v) <= len(comp) - 1)

def optimization_problem(path_input_file, path_output_file):
    input = Input()
    input.decode_input(path_input_file)

    # ----------------------------------------- Gurobi model ---------------------------------------------------
    N, K = input.N, input.K
    depot = 0
    customers = list(range(1, N+1)) # FIXME 1 ?
    nodes = list(range(0, N+1))
    vehicles = list(range(0, K))

    # Create an empty model
    m = g.Model()

    # ---------------------------------------- 
    # Variables 
    # ----------------------------------------
    # x[d,u,v] = 1 if vehicle d goes directly from u to v (u != v)
    x = m.addVars(
        vehicles, nodes, nodes,
        vtype=g.GRB.BINARY,
        name="x"
    )

    # Forbid self-loops: x[d,i,i] = 0
    for d in vehicles:
        for i in nodes:
            x[d, i, i].ub = 0 # Set upper bound to be 0 (should be faster in comparison with constrain == 0)

    # y[d,i] = 1 if vehicle d serves customer i
    y = m.addVars(vehicles, customers, vtype=g.GRB.BINARY, name="y")

    # z[d] = 1 if vehicle d is used (leaves depot)
    z = m.addVars(vehicles, vtype=g.GRB.BINARY, name="z")

    # t[d,i] arrival time of vehicle d to customer i
    # (We do not need depot time explicitly.)
    t = m.addVars(vehicles, customers, vtype=g.GRB.CONTINUOUS, lb=0.0, name="t") # lb - lower bound

    # Big-M (constatnt)
    max_Tout = max(input.T_out) if N > 0 else 0.0
    max_travel = max(max(row) for row in input.T) if N >= 0 else 0.0
    M = max_Tout + max_travel + 1.0  # +1 as a small cushion

    # ---------------------------------------- 
    # Allow lazy constrains + callback variables
    # ---------------------------------------- 
    m.Params.LazyConstraints = 1
    m._x = x
    m._z = z
    m._vehicles = vehicles
    m._nodes = nodes
    m._customers = customers

    # ---------------------------------------- 
    # Constraints 
    # ----------------------------------------
    # (1) Link y to incoming arcs: y[d,i] == sum_u x[d,u,i] 
    # Has van d arrived at node i from aribtrary node? (1/0) 
    for d in vehicles:
        for i in customers:
            m.addConstr(
                y[d, i] == g.quicksum(x[d, u, i] for u in nodes if u != i),
                name=f"link_y_in_{d}_{i}"
            )

    # (2) Each customer served exactly once across all vehicles: sum_d y[d,i] == 1
    for i in customers:
        m.addConstr(
            g.quicksum(y[d, i] for d in vehicles) == 1,
            name=f"serve_once_{i}"
        )

    # (3) If van d arrived to i, it also has to leave it (Flow conservation per vehicle at each customer):
    # sum_u x[d,u,i] == sum_v x[d,i,v]
    for d in vehicles:
        for i in customers:
            m.addConstr(
                g.quicksum(x[d, u, i] for u in nodes if u != i) == g.quicksum(x[d, i, v] for v in nodes if v != i),
                name=f"flow_{d}_{i}"
            )

    # (4) Leave depot at most once; and return if left.
    for d in vehicles:
        leave = g.quicksum(x[d, depot, v] for v in customers)
        ret = g.quicksum(x[d, u, depot] for u in customers)

        # m.addConstr(leave <= 1, name=f"depot_leave_once_{d}") # Redundant since z is binary
        m.addConstr(ret == leave, name=f"depot_return_{d}")

        # Link z: if leaves then z=1
        m.addConstr(leave == z[d], name=f"link_z_{d}")

    # (5) Symmetry breaking: use lower-indexed vehicles first 3 To make the program faster
    for d in range(K - 1):
        m.addConstr(z[d] >= z[d + 1], name=f"symmetry_use_order_{d}")

    # (6) Capacity per vehicle: sum_i s_i * y[d,i] <= Q
    for d in vehicles:
        m.addConstr(
            g.quicksum(input.s[i] * y[d, i] for i in customers) <= input.Q,
            name=f"capacity_{d}"
        )

    # (7) Time windows per served customer:
    # Activate time window only if vehicle d serves customer i.
    # If y[d,i] = 1 → enforce T_in[i] ≤ t[d,i] ≤ T_out[i].
    # If y[d,i] = 0 → Big-M relaxes the constraint (time variable is unconstrained).
    for d in vehicles:
        for i in customers:
            m.addConstr(
                t[d, i] >= input.T_in[i] - M * (1 - y[d, i]),
                name=f"tw_lb_{d}_{i}"
            )
            m.addConstr(
                t[d, i] <= input.T_out[i] + M * (1 - y[d, i]),
                name=f"tw_ub_{d}_{i}"
            )

    # (8) Time precedence along arcs between customers:
    # if x[d,u,v] = 1 then t[v] >= t[u] + T_uv
    for d in vehicles:
        for u in customers:
            for v in customers:
                if u == v:
                    continue
                m.addConstr(
                    t[d, v] >= t[d, u] + input.T[u][v] - M * (1 - x[d, u, v]),
                    name=f"time_prec_{d}_{u}_{v}"
                )

    # (9) Depot -> first customer time feasibility:
    # if x[d,0,v]=1 then t[v] >= T_0v (depart at time 0)
    for d in vehicles:
        for v in customers:
            m.addConstr(
                t[d, v] >= input.T[depot][v] - M * (1 - x[d, depot, v]),
                name=f"time_from_depot_{d}_{v}"
            )

    # ---------------------------------------- 
    # Objective 
    # ----------------------------------------
    travel_cost = g.quicksum(
        input.C[u][v] * x[d, u, v]
        for d in vehicles
        for u in nodes
        for v in nodes
        if u != v
    )
    fixed_cost = input.Gamma * g.quicksum(z[d] for d in vehicles)

    m.setObjective(travel_cost + fixed_cost, g.GRB.MINIMIZE)

    # ---------------------------------------- 
    # Optimize 
    # ----------------------------------------
    m.optimize(subtour_elimination_callback)
    
    write_solution(
        path_output_file,
        m,
        x,
        t,
        z,
        vehicles,
        nodes,
        customers
    )

def main():
    if len(sys.argv) != 3:
        print("Usage: ./your-solver PATH_INPUT_FILE PATH_OUTPUT_FILE")
        sys.exit(1)
    
    path_input_file = sys.argv[1]
    path_output_file = sys.argv[2]

    print("Input file:", path_input_file)
    print("Output file:", path_output_file)

    optimization_problem(path_input_file, path_output_file)


if __name__ == "__main__":
    main()

