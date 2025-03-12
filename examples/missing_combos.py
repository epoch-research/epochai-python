"""
Script to find missing model-task combinations in benchmark runs.

This script identifies which model-task combinations don't have benchmark runs,
helping to identify gaps in benchmark coverage.
"""

from rich.console import Console
from rich.table import Table
from collections import defaultdict
import argparse
from itertools import product
import time

from epochai.airtable.models import MLModel, Task, BenchmarkRun

console = Console()

def fetch_all_data():
    """
    Fetch all required data from Airtable in one place.
    
    Returns:
        Tuple of (runs, models, tasks, model_lookup, task_lookup)
    """
    console.print("[yellow]Fetching data from Airtable...[/]")
    start_time = time.time()
    
    # Fetch all data with a single API call per table
    runs = BenchmarkRun.all(memoize=True)  # This will make a single API call
    models = MLModel.all(memoize=True)     # This will make a single API call
    tasks = Task.all(memoize=True)         # This will make a single API call
    
    # Create lookups
    model_lookup = {model.model_id: model for model in models}
    task_lookup = {task.path: task for task in tasks}
    
    elapsed = time.time() - start_time
    console.print(f"[green]Data fetched in {elapsed:.2f} seconds.[/]")
    
    return runs, models, tasks, model_lookup, task_lookup

def get_missing_combinations(runs, models, tasks):
    """
    Find model-task combinations that don't have benchmark runs.
    
    Only considers:
    - Models that appear in at least one successful benchmark run
    - Only benchmark runs with a status of "Success"
    """
    # Filter runs with status "Success"
    successful_runs = [run for run in runs if run.status == "Success"]
    console.print(f"[green]Found {len(successful_runs)} successful benchmark runs out of {len(runs)} total runs.[/]")
    
    # Create a set of (model_id, task_path) tuples for existing successful combinations
    existing_combinations = {
        (run.model.model_id, run.task.path)
        for run in successful_runs
    }
    
    # Get the set of models that appear in at least one successful benchmark run
    active_models = {run.model.model_id for run in successful_runs}
    console.print(f"[blue]Found {len(active_models)} models with at least one successful benchmark run.[/]")
    
    # Create a set of all possible combinations, but only with models that have
    # been used successfully at least once
    all_combinations = set(product(
        active_models,
        [task.path for task in tasks]
    ))
    
    # Find missing combinations
    missing_combinations = all_combinations - existing_combinations
    
    return missing_combinations

def print_missing_combinations(missing_combinations, model_lookup, task_lookup, group_by=None, model_filter=None, task_filter=None):
    """Print missing model-task combinations."""
    filtered_combinations = missing_combinations
    
    # Apply filters if provided
    if model_filter:
        filtered_combinations = {
            (model_id, task_path) for model_id, task_path in filtered_combinations
            if model_filter.lower() in model_id.lower()
        }
    
    if task_filter:
        filtered_combinations = {
            (model_id, task_path) for model_id, task_path in filtered_combinations
            if task_filter.lower() in task_path.lower()
        }
    
    if not filtered_combinations:
        console.print("[bold red]No missing combinations found with the current filters.[/]")
        return
    
    # Group the results if requested
    if group_by == "model":
        # Group by model
        grouped = defaultdict(list)
        for model_id, task_path in filtered_combinations:
            grouped[model_id].append(task_path)
        
        # Print grouped results
        for model_id, task_paths in sorted(grouped.items()):
            model = model_lookup.get(model_id)
            model_name = model_id
            
            # Get organization info if available
            org_info = ""
            if model and model.model_group and model.model_group.organizations:
                orgs = [org.name for org in model.model_group.organizations]
                org_info = f" ({', '.join(orgs)})"
            
            console.print(f"\n[bold blue]Model: {model_name}{org_info}[/]")
            
            # Create a table for missing tasks
            table = Table(show_header=True, header_style="bold")
            table.add_column("Missing Tasks", style="magenta")
            
            for task_path in sorted(task_paths):
                table.add_row(task_path)
            
            console.print(table)
    
    elif group_by == "task":
        # Group by task
        grouped = defaultdict(list)
        for model_id, task_path in filtered_combinations:
            grouped[task_path].append(model_id)
        
        # Print grouped results
        for task_path, model_ids in sorted(grouped.items()):
            task = task_lookup.get(task_path)
            task_name = task.name if task and task.name else task_path
            
            console.print(f"\n[bold magenta]Task: {task_name} ({task_path})[/]")
            
            # Create a table for missing models
            table = Table(show_header=True, header_style="bold")
            table.add_column("Missing Models", style="blue")
            
            for model_id in sorted(model_ids):
                table.add_row(model_id)
            
            console.print(table)
    
    else:
        # No grouping, just show all missing combinations
        table = Table(show_header=True, header_style="bold", title="Missing Model-Task Combinations")
        table.add_column("Model", style="blue")
        table.add_column("Task", style="magenta")
        
        for model_id, task_path in sorted(filtered_combinations):
            table.add_row(model_id, task_path)
        
        console.print(table)
        console.print(f"Total missing combinations: {len(filtered_combinations)}")

def print_summary(missing_combinations, model_lookup, task_lookup):
    """Print summary statistics about missing combinations."""
    # Count models and tasks with missing combinations
    models_with_missing = {model_id for model_id, _ in missing_combinations}
    tasks_with_missing = {task_path for _, task_path in missing_combinations}
    
    total_models = len(model_lookup)
    total_tasks = len(task_lookup)
    
    # Calculate completion percentage
    all_possible_combinations = total_models * total_tasks
    existing_combinations = all_possible_combinations - len(missing_combinations)
    completion_percentage = (existing_combinations / all_possible_combinations) * 100 if all_possible_combinations > 0 else 0
    
    # Print summary
    console.print("\n[bold]Summary Statistics[/]")
    console.print(f"Total models: {total_models}")
    console.print(f"Total tasks: {total_tasks}")
    console.print(f"Possible combinations: {all_possible_combinations}")
    console.print(f"Missing combinations: {len(missing_combinations)}")
    console.print(f"Completion percentage: {completion_percentage:.2f}%")
    
    # Find top 5 models and tasks with most missing combinations
    model_missing_count = defaultdict(int)
    task_missing_count = defaultdict(int)
    
    for model_id, task_path in missing_combinations:
        model_missing_count[model_id] += 1
        task_missing_count[task_path] += 1
    
    # Print models with most missing combinations
    console.print("\n[bold blue]Top 5 Models with Most Missing Tasks[/]")
    for model_id, count in sorted(model_missing_count.items(), key=lambda x: x[1], reverse=True)[:5]:
        completion = ((total_tasks - count) / total_tasks) * 100
        console.print(f"{model_id}: Missing {count}/{total_tasks} tasks ({completion:.2f}% complete)")
    
    # Print tasks with most missing combinations
    console.print("\n[bold magenta]Top 5 Tasks with Most Missing Models[/]")
    for task_path, count in sorted(task_missing_count.items(), key=lambda x: x[1], reverse=True)[:5]:
        completion = ((total_models - count) / total_models) * 100
        task = task_lookup.get(task_path)
        task_name = task.name if task and task.name else task_path
        console.print(f"{task_name} ({task_path}): Missing {count}/{total_models} models ({completion:.2f}% complete)")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Find missing model-task combinations in benchmark runs.')
    parser.add_argument('--group-by', choices=['model', 'task'], default='task',
                        help='Group results by model or task (default: task)')
    parser.add_argument('--model-filter', help='Filter by model ID (case-insensitive substring match)')
    parser.add_argument('--task-filter', help='Filter by task path (case-insensitive substring match)')
    parser.add_argument('--summary', action='store_true', help='Show summary statistics')
    
    args = parser.parse_args()
    
    # Fetch all data at once to minimize API calls
    runs, models, tasks, model_lookup, task_lookup = fetch_all_data()
    
    # Find missing combinations
    missing_combinations = get_missing_combinations(runs, models, tasks)
    
    # Print summary if requested
    if args.summary:
        print_summary(missing_combinations, model_lookup, task_lookup)
    
    # Print missing combinations
    print_missing_combinations(
        missing_combinations, 
        model_lookup, 
        task_lookup, 
        group_by=args.group_by,
        model_filter=args.model_filter,
        task_filter=args.task_filter
    )

if __name__ == "__main__":
    main()
