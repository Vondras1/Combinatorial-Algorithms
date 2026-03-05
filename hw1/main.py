#!/usr/bin/env python3
import numpy as np
import gurobipy as g  # import Gurobi
import sys

class Input():
    def __init__(self):
        self.R = None
        self.r_rows = []
        self.r_cols = []
    
    def decode_input(self, file_name):
        input_lines = self.load_input(file_name)
        self.R = int(input_lines[0][0])
        if self.R >= 1:
            for rook in input_lines[1:]:
                # each rook line is like ["a1"]
                letter = rook[0][0]
                number = int(rook[0][1])

                # convert to 0-based indices for x[0..7, 0..7]
                self.r_rows.append(number-1)
                self.r_cols.append(self._letter_to_number(letter))
              
    def load_input(self, file_name):
        with open(file_name, "r") as file:
            input_lines = []
            for line in file:
                row = []
                for token in line.strip().split():
                    row.append(token)
                input_lines.append(row)
        return input_lines

    def _letter_to_number(self, letter):
        return ord(letter) - ord('a')

if __name__ == "__main__":
    # sys.argv[0] je název skriptu, argumenty začínají od indexu 1
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]


    input = Input()
    input.decode_input(file_name=arg1)

    # Constants
    num_of_rows = 8
    num_of_columns = 8

    # Create an empty model
    m = g.Model()

    # ---------------------------------------- 
    # Variables 
    # ----------------------------------------
    # x[8,8] = 1 if knight is at possition i, j
    x = m.addVars(
        num_of_rows, num_of_columns,
        vtype=g.GRB.BINARY,
        name="x"
    )

    # ---------------------------------------- 
    # Constraints 
    # ----------------------------------------
    # (1) Block rows and columns occupied by rooks
    for rooks_idx in range(input.R):
        rr = input.r_rows[rooks_idx]
        cc = input.r_cols[rooks_idx]

        # whole column cc
        for r in range(num_of_rows):
            x[r, cc].UB = 0 # Instead of adding a constraint x[r, cc] == 0, add directly an upper bound of 0 <--- much more efficient

        # whole row rr
        for c in range(num_of_columns):
            x[rr, c].UB = 0

    # # # !!! Working but too slow (because of many constraints) !!!
    # # (1) Block rows and columns occupied by rooks
    # for rooks_idx in range(input.R):
    #     rr = input.r_rows[rooks_idx]
    #     cc = input.r_cols[rooks_idx]

    #     # whole column cc
    #     for r in range(num_of_rows):
    #         m.addConstr(x[r, cc] == 0)

    #     # whole row rr
    #     for c in range(num_of_columns):
    #         m.addConstr(x[rr, c] == 0)

    # (2) No two knights can attack each other
    moves = [(1, 2),(2, 1),(-1, 2),(-2, 1),(1, -2),(2, -1),(-1, -2),(-2, -1)]
    for row in range(num_of_rows):
        for col in range(num_of_columns):
            for dr, dc in moves:
                r2 = row + dr
                c2 = col + dc
                if 0 <= r2 < num_of_rows and 0 <= c2 < num_of_columns:
                    if (row, col) < (r2, c2):
                        m.addConstr(x[row, col] + x[r2, c2] <= 1)

    # ---------------------------------------- 
    # Objective
    # ---------------------------------------- 
    m.setObjective(
        g.quicksum(x[i, j] for i in range(num_of_rows) for j in range(num_of_columns)), 
        g.GRB.MAXIMIZE
    )

    # ---------------------------------------- 
    # Optimize 
    # ----------------------------------------
    m.optimize()

    # ----------------------------------------
    # Output (format: N then N lines with positions)
    # ----------------------------------------
    if m.Status == g.GRB.OPTIMAL:
        knights = []
        for i in range(num_of_rows):
            for j in range(num_of_columns):
                if x[i, j].X == 1:  # if there's a knight at (i, j)
                    # Convert (row=i, col=j) -> "a1" style
                    pos = f"{chr(ord('a') + j)}{i + 1}"
                    knights.append(pos)

        N = len(knights)

        with open(arg2, "w") as f:
            f.write(f"{N}\n")
            for p in knights:
                f.write(p + "\n")

        # # Also print to terminal for debugging
        # print(N)
        # for p in knights:
        #     print(p)
    else:
        raise RuntimeError(f"No optimal solution found. Gurobi status: {m.Status}")