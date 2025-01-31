from pyairtable.orm import Model, fields as F
from epochai.airtable.client import AIRTABLE_TOKEN, BASE_ID
from epochai.airtable.models import MLModel, Task, Score, Organization, BenchmarkRun
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.text import Text

console = Console()

# Define consistent colors for different types of objects
COLORS = {
    "MODEL": "blue",
    "ORG": "yellow",
    "TASK": "magenta",
    "SCORE": "green",
    "ERROR": "red",
    "DEVELOPER": "cyan",
}

def print_model_info(model_id: str):
    model = MLModel.first(formula=f"id='{model_id}'")
    if not model:
        raise ValueError(f"Model {model_id} not found")

    # Print basic model info
    console.print(f"\n[bold {COLORS['MODEL']}]Model: {model.model_id}[/]")
    
    # Print organizations
    console.print(f"\n[{COLORS['ORG']}]Organizations:[/]")
    for org in model.model_group.organizations:
        console.print(f"  • {org.name}")
    
    # Print HF developer if available
    if model.hf_developer:
        console.print(f"\n[{COLORS['DEVELOPER']}]Hugging Face Developer: {model.hf_developer}[/]")

    # Print benchmark runs
    for run in model.benchmark_runs:
        console.print(f"\n[bold {COLORS['TASK']}]Task: {run.task.path}[/]")
        print(f"Log viewer: {run.log_viewer}")  # Using plain print for URLs
        
        # Create a table just for scores
        table = Table(show_header=True, header_style="bold", title="Scores")
        table.add_column("Scorer", style=COLORS['ORG'])
        table.add_column("Score", justify="right", style=COLORS['SCORE'])
        table.add_column("Std Error", justify="right", style=COLORS['ERROR'])
        
        for score in run.scores:
            table.add_row(
                score.scorer,
                f"{score.mean:.3f}",
                f"±{score.stderr:.3f}"
            )
        
        console.print(table)
        print("─" * 80)  # Separator between runs

def print_high_scores(task_path: str, scores: list[Score], scorer: str):
    task = Task.first(formula=f"path='{task_path}'")
    if not task:
        raise ValueError(f"Task {task_path} not found")
    
    # Get all scores for the task and scorer
    task_scores = [
        score for score in scores
        if score.benchmark_run.task.path == task_path and score.scorer == scorer
    ]
    
    # Sort by mean score in descending order and take top 10
    top_scores = sorted(task_scores, key=lambda x: x.mean, reverse=True)[:10]
    
    # Create a rich table for top scores
    table = Table(title=f"\nTop 10 Scores for {task_path} ({scorer})", show_header=True, header_style="bold")
    table.add_column("Model ID", style=COLORS['MODEL'])
    table.add_column("Score", justify="right", style=COLORS['SCORE'])
    table.add_column("Std Error", justify="right", style=COLORS['ERROR'])
    
    for score in top_scores:
        run = score.benchmark_run
        table.add_row(
            run.model.model_id,
            f"{score.mean:.3f}",
            f"±{score.stderr:.3f}"
        )
    
    console.print(table)

def main():
    # Get everything at the start to minimize API calls
    scores = Score.all(memoize=True)
    runs = BenchmarkRun.all(memoize=True)
    models = MLModel.all(memoize=True)
    tasks = Task.all(memoize=True)
    organizations = Organization.all(memoize=True)

    print_model_info("claude-3-5-sonnet-20240620")
    print_high_scores("bench.task.hendrycks_math.hendrycks_math_lvl_5", scores, "model_graded_equiv")

if __name__ == "__main__":
    main()
