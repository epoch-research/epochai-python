from rich.console import Console
from rich.table import Table
from datetime import datetime
from collections import defaultdict

from epochai.airtable.models import MLModel, Score, Task

console = Console()

# Define colors for consistent styling
COLORS = {
    "MODEL": "blue",
    "ORG": "yellow",
    "TASK": "magenta",
    "SCORE": "green",
    "ERROR": "red",
    "DATE": "grey70",
    "ACCESSIBILITY": "orange",
    "COMPUTE": "purple",
}

def get_reasoning_models():
    """Fetch and return all reasoning-focused models (O1, O3, and Deepseek-R1)."""
    models = MLModel.all(memoize=True)
    reasoning_models = [
        model for model in models 
        if model.model_id.startswith(("o1-", "o3-")) or model.model_id == "DeepSeek-R1"
    ]
    # remove -preview models
    reasoning_models = [
        model for model in reasoning_models 
        if not "preview" in model.model_id
    ]
    return sorted(reasoning_models, key=lambda x: x.release_date if x.release_date else datetime.min)

def print_model_comparison(models: list[MLModel], tasks: list[Task], task_scorers: dict[str, str]):
    """
    Compare reasoning models across specific benchmark tasks.
    
    Args:
        models: List of models to compare
        tasks: List of tasks to evaluate
        task_scorers: Dictionary mapping task paths to their appropriate scorers
    """
    scores = Score.all(memoize=True)
    
    # Collect scores for each model and task
    model_scores = defaultdict(dict)
    for score in scores:
        if score.benchmark_run.model.model_id in [m.model_id for m in models]:
            task_path = score.benchmark_run.task.path
            if task_path in task_scorers and score.scorer == task_scorers[task_path]:
                model_scores[score.benchmark_run.model.model_id][task_path] = score

    # Create comparison table
    table = Table(
        title="\nReasoning Model Performance Comparison",
        show_header=True,
        header_style="bold"
    )
    
    table.add_column("Model ID", style=COLORS['MODEL'])
    table.add_column("Release Date", style=COLORS['DATE'])
    
    # Add columns for each task
    for task in tasks:
        scorer = task_scorers[task.path]
        table.add_column(f"{task.name or task.path}\n[dim]({scorer})[/]", style=COLORS['SCORE'])
    
    # Add rows for each model
    for model in models:
        row = [model.model_id]
        row.append(model.release_date.strftime("%Y-%m-%d") if model.release_date else "N/A")
        
        # Check if model has any scores before adding row
        has_scores = False
        scores_for_tasks = []
        
        # Add scores for each task
        for task in tasks:
            if task.path in model_scores[model.model_id]:
                score = model_scores[model.model_id][task.path]
                scores_for_tasks.append(
                    f"{score.mean:.3f} [dim {COLORS['ERROR']}]±{score.stderr:.3f}[/]"
                )
                has_scores = True
            else:
                scores_for_tasks.append("N/A")
        
        # Only add row if model has at least one score
        if has_scores:
            row.extend(scores_for_tasks)
            table.add_row(*row)
    
    console.print(table)

def main():
    # Get reasoning models
    reasoning_models = get_reasoning_models()
    
    # Print basic info about reasoning models
    console.print("\n[bold]Found 'reasoning'-models:[/]")
    for model in reasoning_models:
        console.print(f"• [blue]{model.model_id}[/]")
    
    # Get Task objects for interesting tasks
    tasks = Task.all(memoize=True)
    interesting_tasks = [
        task for task in tasks
        if task.path in [
            "bench.task.hendrycks_math.hendrycks_math_lvl_5",
            "bench.task.gpqa.gpqa_diamond",
        ]
    ]
    
    # Define which scorer to use for each task
    task_scorers = {
        "bench.task.hendrycks_math.hendrycks_math_lvl_5": "model_graded_equiv",
        "bench.task.gpqa.gpqa_diamond": "choice"
    }
    
    print_model_comparison(
        reasoning_models, 
        interesting_tasks,
        task_scorers=task_scorers
    )

if __name__ == "__main__":
    main() 