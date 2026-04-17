#!/usr/bin/env python3
from dataclasses import dataclass
from typing import List, Optional
import sys
import numpy as np
from collections import deque

@dataclass
class Customer:
    min_reviews: int
    max_reviews: int
    known_products: List[int]

class LoadInput:
    def __init__(self, num_customers: int, num_products: int, customers: List[Customer], product_demands: List[int]) -> None:
        self.num_customers = num_customers
        self.num_products = num_products
        self.customers = customers
        self.product_demands = product_demands

    @classmethod
    def from_file(cls, input_path: str) -> "LoadInput":
        with open(input_path, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]

        first_line = lines[0].split()
        num_customers = int(first_line[0])
        num_products = int(first_line[1])

        customers: List[Customer] = []

        for i in range(1, 1 + num_customers):
            parts = list(map(int, lines[i].split()))
            min_reviews = parts[0]
            max_reviews = parts[1]
            known_products = parts[2:]

            customers.append(Customer(
                min_reviews=min_reviews,
                max_reviews=max_reviews,
                known_products=known_products
            ))

        parts = lines[1 + num_customers].split()
        product_demands = [int(x) for x in parts]

        return cls(
            num_customers=num_customers,
            num_products=num_products,
            customers=customers,
            product_demands=product_demands
        )

class Edge:
    def __init__(self, start_node: int, end_node: int, capacity: int, lower_bound: int = 0) -> None:
        self.start_node = start_node
        self.end_node = end_node

        # Capacity in transformed residual network (after lower bound transformation)
        self.capacity = capacity

        # Original lower bound (for flow reconstruction)
        self.lower_bound = lower_bound

        # Flow in transformed network
        self.flow = 0

        self.reverse_edge: Optional["Edge"] = None

        # Whether this is a real/original edge, useful for output/reconstruction
        self.is_original = False

    @property
    def residual_capacity(self) -> int:
        return self.capacity - self.flow

class Graph:
    def __init__(self, num_customers, num_products):
        self.num_customers = num_customers
        self.num_products = num_products
        
        self.num_original_nodes = num_customers + num_products + 1 + 1 # +1 for source, +1 for sink
        self.num_nodes = self.num_original_nodes + 2 # +2 for super_source and super_sink in lower-bound transformation
        self.adj_list = [[] for _ in range(self.num_nodes)]

        self.source = 0
        self.sink = self.num_original_nodes - 1

        # After lower-bound transformation we append
        # super_source, super_sink for flow initialization
        self.super_source = self.num_original_nodes
        self.super_sink = self.num_original_nodes + 1

        # Node balance for lower-bound feasibility transformation
        self.balance = [0] * self.num_original_nodes

        self.customer_product_edges: List[List[Edge]] = [[] for _ in range(num_customers)]

        # Super source and super sink edges for lower-bound transformation
        self.ss_edges: List[Edge] = []
        self.ts_edge: Optional[Edge] = None

    def get_customer_i(self, i):
        return 1 + i
    
    def get_product_j(self, j):
        return 1 + self.num_customers + j

    def create_dual_edge(self, start_node: int, end_node: int, capacity: int, lower_bound: int = 0, is_original: bool = False) -> Edge:
        forward_edge = Edge(start_node, end_node, capacity, lower_bound)
        reverse_edge = Edge(end_node, start_node, 0, 0)

        forward_edge.reverse_edge = reverse_edge
        reverse_edge.reverse_edge = forward_edge

        forward_edge.is_original = is_original
        reverse_edge.is_original = False

        self.adj_list[start_node].append(forward_edge)
        self.adj_list[end_node].append(reverse_edge)

        return forward_edge

    def create_bounded_edge(self, start_node: int, end_node: int, lower_bound: int, upper_bound: int, is_original: bool = True) -> Edge:
        if lower_bound < 0 or upper_bound < 0 or lower_bound > upper_bound:
            raise ValueError(f"Invalid bounds on edge {start_node}->{end_node}: [{lower_bound}, {upper_bound}]")

        # Transformed capacity is upper - lower
        edge = self.create_dual_edge(
            start_node=start_node,
            end_node=end_node,
            capacity=upper_bound - lower_bound,
            lower_bound=lower_bound,
            is_original=is_original
        )

        # Update balance for lower-bound reduction
        # For edge u->v with lower l:
        # balance[u] -= l
        # balance[v] += l
        self.balance[start_node] -= lower_bound
        self.balance[end_node] += lower_bound

        return edge
    
    def add_customer_edges(self, customer: Customer, customer_index: int) -> None:
        customer_node = self.get_customer_i(customer_index)
        self.create_bounded_edge(
            self.source,
            customer_node,
            customer.min_reviews,
            customer.max_reviews,
            is_original=True
        )

    def add_customer_product_edges(self, customer: Customer, customer_index: int) -> None:
        customer_node = self.get_customer_i(customer_index)

        for product in customer.known_products:
            product_index = product - 1
            if not (0 <= product_index < self.num_products):
                raise ValueError(f"Invalid product id {product} for customer {customer_index + 1}")

            product_node = self.get_product_j(product_index)
            edge = self.create_bounded_edge(
                customer_node,
                product_node,
                0,
                1,
                is_original=True
            )
            self.customer_product_edges[customer_index].append(edge)

    def add_product_edges(self, product_demand: int, product_index: int) -> None:
        product_node = self.get_product_j(product_index)

        # Safe upper bound: at most all customers can review the product
        self.create_bounded_edge(
            product_node,
            self.sink,
            product_demand,
            self.num_customers,
            is_original=True
        )

    def build_graph(self, customers: List[Customer], product_demands: List[int]):
        for i, customer in enumerate(customers):
            self.add_customer_edges(customer, i)
            self.add_customer_product_edges(customer, i)
        
        for j, demand in enumerate(product_demands):
            self.add_product_edges(demand, j)

    def add_feasibility_transformation(self) -> int:
        """
        Adds:
          - edge sink -> source with large capacity
          - super_source / super_sink edges from balances

        Returns total demand that must be sent from super_source.
        """
        # Large enough capacity for sink -> source
        big_capacity = self.num_customers * self.num_products

        self.ts_edge = self.create_dual_edge(
            self.sink,
            self.source,
            big_capacity,
            lower_bound=0,
            is_original=False
        )

        total_positive_balance = 0

        for node in range(self.num_original_nodes):
            if self.balance[node] > 0:
                edge = self.create_dual_edge(
                    self.super_source,
                    node,
                    self.balance[node],
                    lower_bound=0,
                    is_original=False
                )
                self.ss_edges.append(edge)
                total_positive_balance += self.balance[node]

            elif self.balance[node] < 0:
                self.create_dual_edge(
                    node,
                    self.super_sink,
                    -self.balance[node],
                    lower_bound=0,
                    is_original=False
                )
                
        return total_positive_balance

    def original_real_flow(self, edge: Edge) -> int:
        """
        For original bounded edge:
            real flow = lower_bound + transformed flow
        """
        return edge.lower_bound + edge.flow

    def augment_along_path(self, path_edges: List[Edge], delta: int) -> None:
        for edge in path_edges:
            edge.flow += delta
            edge.reverse_edge.flow -= delta

    def all_super_source_edges_saturated(self) -> bool:
        return all(edge.residual_capacity == 0 for edge in self.ss_edges)

# # TODO Maybe try first to use classic Ford-Fulkerson with DFS and only if that is too slow, implement Edmonds-Karp with BFS
def bfs_find_augmenting_path(graph: Graph, start_node: int, target_node: int, active_nodes: int) -> tuple[List[Edge], int]:
    parent_edge: List[Optional[Edge]] = [None] * active_nodes
    visited = [False] * active_nodes

    queue = deque()
    queue.append(start_node)
    visited[start_node] = True

    while queue:
        current_node = queue.popleft()

        for edge in graph.adj_list[current_node]:
            next_node = edge.end_node

            # Skip super source/sink nodes in the second phase of Edmonds-Karp
            if next_node >= active_nodes:
                continue

            # Skip visited nodes and edges with no residual capacity
            if visited[next_node]:
                continue

            # Only consider edges with available residual capacity
            if edge.residual_capacity <= 0:
                continue

            visited[next_node] = True
            parent_edge[next_node] = edge
            queue.append(next_node)

            if next_node == target_node:
                queue.clear()
                break

    if not visited[target_node]:
        return [], 0

    # Reconstruct path and find bottleneck(= max amount of flow you can send along the whole path)
    path_edges: List[Edge] = []
    bottleneck = float("inf")
    current_node = target_node

    while current_node != start_node:
        edge = parent_edge[current_node]
        if edge is None:
            return [], 0

        path_edges.append(edge)
        bottleneck = min(bottleneck, edge.residual_capacity)
        current_node = edge.start_node

    path_edges.reverse()
    return path_edges, int(bottleneck)


def edmonds_karp(graph: Graph, start_node: int, target_node: int, active_nodes: int) -> int:
    max_flow = 0

    while True:
        path_edges, delta = bfs_find_augmenting_path(graph, start_node, target_node, active_nodes)

        if delta == 0:
            break

        graph.augment_along_path(path_edges, delta)
        max_flow += delta

    return max_flow

def save_result(graph: Graph, customers: List[Customer], output_path: str, feasible: bool) -> None:
    with open(output_path, "w", encoding="utf-8") as file:
        if not feasible:
            file.write("-1\n")
            return

        first_product_node = 1 + graph.num_customers
        last_product_node = first_product_node + graph.num_products - 1

        for customer_index in range(graph.num_customers):
            customer_node = graph.get_customer_i(customer_index)
            assigned_products = []

            for edge in graph.adj_list[customer_node]:
                if edge.is_original and first_product_node <= edge.end_node <= last_product_node and graph.original_real_flow(edge) > 0:
                    product_number = edge.end_node - first_product_node + 1
                    assigned_products.append(product_number)

            assigned_products.sort()
            file.write(" ".join(map(str, assigned_products)) + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: ./your-solver PATH_INPUT_FILE PATH_OUTPUT_FILE")
        sys.exit(1)
    
    path_input_file = sys.argv[1]
    path_output_file = sys.argv[2]

    print("Input file:", path_input_file)
    print("Output file:", path_output_file)

    # Load data from input file
    data = LoadInput.from_file(path_input_file)

    print("Number of customers:", data.num_customers)
    print("Number of products:", data.num_products)
    print("Known products for customer 0:", data.customers[0].known_products)
    print("Product demands:", data.product_demands)

    # Build the graph
    graph = Graph(data.num_customers, data.num_products)
    try:
        graph.build_graph(data.customers, data.product_demands)
    except ValueError:
        # Invalid input, treat as infeasible
        save_result(graph, data.customers, path_output_file, feasible=False)
        print("Feasible: False")
        sys.exit(0)

    # Phase 1: find initial feasible flow with lower bounds
    graph.add_feasibility_transformation()
    edmonds_karp(
        graph,
        graph.super_source,
        graph.super_sink,
        graph.num_nodes
    )

    feasible = graph.all_super_source_edges_saturated()

    if feasible:
        # Phase 2: maximize flow in the original graph only
        max_flow = edmonds_karp(
            graph,
            graph.source,
            graph.sink,
            graph.num_original_nodes
        )
    else:
        max_flow = 0

    save_result(graph, data.customers, path_output_file, feasible)

    print("Feasible:", feasible)
    print("Additional max flow:", max_flow)
