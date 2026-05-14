#!/usr/bin/env python3
from typing import List, Optional
import sys

# Run command: 

class Task:
    def __init__(self, index, release_date, processing_time, deadline):
        self.index = index
        self.release_date = release_date
        self.processing_time = processing_time
        self.deadline = deadline

class LoadInput:
    def __init__(self, num_tasks: int, tasks: List[Task]) -> None:
        self.num_tasks = num_tasks
        self.tasks = tasks

    @classmethod
    def from_file(cls, input_path: str) -> "LoadInput":
        with open(input_path, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]

        first_line = lines[0].split()
        num_tasks = int(first_line[0])

        tasks: List[Task] = []

        for i in range(1, 1 + num_tasks):
            parts = list(map(int, lines[i].split()))
            processing_time = parts[0]
            release_date = parts[1]
            deadline = parts[2]

            tasks.append(Task(
                index=i - 1,
                processing_time=processing_time,
                release_date=release_date,
                deadline=deadline,
            ))
            
        return cls(num_tasks=num_tasks, tasks=tasks)
    
def search(scheduled, unscheduled, current_time, best_objective, best_schedule):
    # Complete schedule
    if not unscheduled:
        if current_time < best_objective:
            return current_time, scheduled.copy()
        return best_objective, best_schedule

    # Rule 1: missed deadline
    for task in unscheduled:
        earliest_completion = max(current_time, task.release_date) + task.processing_time
        if earliest_completion > task.deadline:
            return best_objective, best_schedule

    # Rule 2: lower bound
    min_release = min(task.release_date for task in unscheduled)
    remaining_processing = sum(task.processing_time for task in unscheduled)
    lower_bound = max(current_time, min_release) + remaining_processing

    if lower_bound >= best_objective:
        return best_objective, best_schedule

    # Rule 3: decomposition / waiting
    if current_time <= min_release:
        # TODO How to use the information that the schedule build into this moment is optimal? We do not have to care about others branches
        current_time = min_release

    # Branch over possible next tasks
    for task in unscheduled:
        start_time = max(current_time, task.release_date)
        completion_time = start_time + task.processing_time

        if completion_time > task.deadline:
            continue

        if completion_time >= best_objective:
            continue

        new_scheduled = scheduled + [(task, start_time)]
        new_unscheduled = [t for t in unscheduled if t is not task]

        best_objective, best_schedule = search(
            new_scheduled,
            new_unscheduled,
            completion_time,
            best_objective,
            best_schedule,
        )

    return best_objective, best_schedule

def bartley(tasks):
    best_objective, best_schedule = search(
        scheduled=[],
        unscheduled=tasks,
        current_time=0,
        best_objective=float("inf"),
        best_schedule=None,
    )

    if best_schedule is None:
        return [-1]

    starts = [0] * len(tasks)

    for task, start_time in best_schedule:
        starts[task.index] = start_time

    return starts

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

    print("Number of tasks:", data.num_tasks)

    starts = bartley(data.tasks)

    print(starts)
    with open(path_output_file, "w", encoding="utf-8") as file:
        for start in starts:
            file.write(f"{start}\n")
