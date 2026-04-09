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


def write_solution(filename, m, x, t, z, vehicles, nodes):
    if m.status != g.GRB.OPTIMAL:
        with open(filename, "w") as f:
            f.write("-1\n")
        return
    # if m.SolCount == 0:
    #     with open(filename, "w") as f:
    #         f.write("-1\n")
    #     return

    used_vehicles = [d for d in vehicles if z[d].X > 0.5]

    with open(filename, "w") as f:
        # první řádek
        f.write(f"{m.objVal} {len(used_vehicles)}\n")

        # pro každé použité vozidlo rekonstruuj trasu
        for d in used_vehicles:
            route = []
            current = 0  # start at depot

            visited = set()
            while True:
                next_node = None
                for v in nodes:
                    if v != current and x[d, current, v].X > 0.5:
                        next_node = v
                        break

                if next_node is None or next_node == 0:
                    break

                if next_node in visited:
                    raise RuntimeError(f"Cycle detected while reconstructing route for vehicle {d}")

                visited.add(next_node)
                route.append(next_node)
                current = next_node

            # výpis jedné trasy
            f.write(str(len(route)))
            for customer in route:
                arrival_time = int(round(t[d, customer].X))
                f.write(f" {customer} {arrival_time}")
            f.write("\n")


def optimization_problem(path_input_file, path_output_file):
    # ---------------------------------------- 
    # Load input
    # ---------------------------------------- 
    inp = Input()
    inp.decode_input(path_input_file)

    # ---------------------------------------- 
    # Define constants 
    # ----------------------------------------
    N = inp.N
    K = inp.K
    # s = inp.s
    # T_in = inp.T_in
    # T_out = inp.T_out
    # T = inp.T
    # C = inp.C
    # Gamma = inp.Gamma
    # Q = inp.Q
    depot = 0
    customers = list(range(1, N+1))
    nodes = list(range(0, N+1))
    vehicles = list(range(0, K))
    
    # Big-M
    max_Tout = max(inp.T_out) if N > 0 else 0.0
    max_travel = max(max(row) for row in inp.T) if N >= 0 else 0.0
    M = max_Tout + max_travel + 1.0  # +1 as a small cushion

    # ---------------------------------------- 
    # Create an empty model
    # ---------------------------------------- 
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
    for d in vehicles:
        for i in nodes:
            x[d, i, i].ub = 0 # Set upper bound to be 0 (should be faster in comparison with constrain == 0)
    
    # t[d,i] arrival time of vehicle d to customer i
    t = m.addVars(vehicles, customers, vtype=g.GRB.CONTINUOUS, lb=0.0, name="t") # lb - lower bound

    # z[d] if vehicle d is used
    z = m.addVars(vehicles, vtype=g.GRB.BINARY, name="z")

    # ---------------------------------------- 
    # Objective 
    # ----------------------------------------
    travel_cost = g.quicksum(
        inp.C[u, v] * x[d, u, v]
        for d in vehicles
        for u in nodes
        for v in nodes
        if u != v
    )
    fixed_cost = inp.Gamma * g.quicksum(z[d] for d in vehicles)

    m.setObjective(travel_cost + fixed_cost, g.GRB.MINIMIZE)

    # ---------------------------------------- 
    # Constraints
    # ----------------------------------------
    # 1) Each customer served exactly once: sum_d(sum_i(x[d,i,j] == 1) for every j
    for j in customers:
        m.addConstr(
            g.quicksum(x[d, i, j] for d in vehicles for i in nodes if i != j) == 1,
            name=f"entre_once_{j}"
        )

    # 2) If van d arrived to i, it also has to leave it (Flow conservation per vehicle at each customer):
    # sum_u x[d,u,i] == sum_v x[d,i,v]
    for d in vehicles:
        for i in customers:
            m.addConstr(
                g.quicksum(x[d, u, i] for u in nodes if u != i) == g.quicksum(x[d, i, v] for v in nodes if v != i),
                name=f"flow_{d}_{i}"
            )
    
    # 3) Each van leaves depot at most once and return if left
    for d in vehicles:
        leave = g.quicksum(x[d, depot, v] for v in customers)
        ret = g.quicksum(x[d, u, depot] for u in customers)

        m.addConstr(leave <= 1, name=f"depot_leave_once_{d}")
        m.addConstr(ret == leave, name=f"depot_return_{d}")

        # Link z: if leaves then z=1
        m.addConstr(leave == z[d], name=f"link_z_{d}")

    # 4) Capacity per vehicle
    for d in vehicles:
        m.addConstr(
            g.quicksum(inp.s[i] * g.quicksum(x[d,i,j] for j in nodes if j != i) for i in customers) <= inp.Q * z[d],
            name=f"capacity_{d}"
        )

    # 5) Time windows per served customer:
    # Activate time window only if vehicle d serves customer i.
    for d in vehicles:
        for i in customers:
            served_by_d = g.quicksum(x[d, j, i] for j in nodes if j != i)
            m.addConstr(
                t[d,i] >= inp.T_in[i] - M*(1 - served_by_d),
                name=f"tw_lb_{d}_{i}"
            )
            m.addConstr(
                t[d,i] <= inp.T_out[i] + M*(1 - served_by_d),
                name=f"tw_ub_{d}_{i}"
            )

    # 6.1) Time precedence along arcs between customers:
    for d in vehicles:
        for u in customers:
            for v in customers:
                if u != v:
                    m.addConstr(
                        t[d,u] + inp.T[u,v] <= t[d,v] + M*(1-x[d,u,v]),
                        name=f"time_arc_{d}_{u}_{v}"
                    )
            
            # 6.2) Time precedence between depot and first customers:
            m.addConstr(
                t[d, u] >= inp.T[depot, u] - M * (1 - x[d, depot, u]),
                name=f"time_from_depot_{d}_{u}"
            )

    m.optimize()
    write_solution(path_output_file, m, x, t, z, vehicles, nodes)

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