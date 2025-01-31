from rich.console import Console
from rich.table import Table

from epochai.airtable.models import MLModel, Task, Score, Organization, BenchmarkRun

console = Console()

# Define consistent colors for different types of objects
COLORS = {
    "MODEL": "blue",
    "ORG": "yellow",
    "TASK": "magenta",
    "SCORE": "green",
    "ERROR": "red",
    "DEVELOPER": "cyan",
    "DATE": "grey70",
}


def print_model_info(model_id: str):
    model = MLModel.first(formula=f"id='{model_id}'")
    if not model:
        raise ValueError(f"Model {model_id} not found")

    # Print basic model info
    console.print(f"\n[bold {COLORS['MODEL']}]Model: {model.model_id}[/]")

    # Print release date if available
    if model.release_date:
        console.print(f"Released: [{COLORS['DATE']}]{model.release_date.strftime('%B %d, %Y')}[/]")
    else:
        console.print("Release date not available")

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
    table = Table(title=f"\nTop 10 Scores for {task_path} ({scorer})", show_header=True,
                  header_style="bold")
    table.add_column("Model ID", style=COLORS['MODEL'])
    table.add_column("Release Date", style=COLORS['DATE'])  # Updated color style
    table.add_column("Score", justify="right", style=COLORS['SCORE'])
    table.add_column("Std Error", justify="right", style=COLORS['ERROR'])

    for score in top_scores:
        run = score.benchmark_run
        release_date = run.model.release_date
        date_str = release_date.strftime("%Y-%m-%d") if release_date else "N/A"
        table.add_row(
            run.model.model_id,
            date_str,
            f"{score.mean:.3f}",
            f"±{score.stderr:.3f}"
        )

    console.print(table)


def print_performance_timeline(task_path: str, scores: list[Score], scorer: str, start_date=None,
                               end_date=None):
    """Show how the best performance on a task evolved over time."""
    # Get all scores for the task and scorer
    task_scores = [
        score for score in scores
        if score.benchmark_run.task.path == task_path
           and score.scorer == scorer
    ]

    if not task_scores:
        console.print(f"[{COLORS['ERROR']}]No scores found for {task_path} with {scorer}[/]")
        return

    # Count and filter models without release dates
    scores_without_dates = [s for s in task_scores if not s.benchmark_run.model.release_date]
    dated_scores = [s for s in task_scores if s.benchmark_run.model.release_date]

    if scores_without_dates:
        console.print(
            f"[{COLORS['ERROR']}]Warning: {len(scores_without_dates)} models were excluded due to missing release dates[/]")
        # Optionally show the excluded models
        excluded_models = {s.benchmark_run.model.model_id for s in scores_without_dates}
        console.print(f"[{COLORS['ERROR']}]Excluded models: {', '.join(excluded_models)}[/]")

    if not dated_scores:
        console.print(f"[{COLORS['ERROR']}]No models with release dates found[/]")
        return

    # Sort scores by release date
    dated_scores = sorted(
        dated_scores,
        key=lambda x: x.benchmark_run.model.release_date
    )

    # Rest of the function remains the same
    best_score = float('-inf')
    improvements = []

    for score in dated_scores:
        if score.mean > best_score:
            best_score = score.mean
            improvements.append({
                'date': score.benchmark_run.model.release_date,
                'model': score.benchmark_run.model.model_id,
                'score': score.mean,
                'stderr': score.stderr
            })

    # Create a table showing improvements over time
    table = Table(
        title=f"\nBest Model Timeline for {task_path} ({scorer})",
        show_header=True,
        header_style="bold"
    )
    table.add_column("Date", style=COLORS['DATE'])
    table.add_column("Model ID", style=COLORS['MODEL'])
    table.add_column("Score", justify="right", style=COLORS['SCORE'])
    table.add_column("Std Error", justify="right", style=COLORS['ERROR'])

    for imp in improvements:
        table.add_row(
            imp['date'].strftime("%Y-%m-%d"),
            imp['model'],
            f"{imp['score']:.3f}",
            f"±{imp['stderr']:.3f}"
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
    print_high_scores("bench.task.hendrycks_math.hendrycks_math_lvl_5", scores,
                      "model_graded_equiv")
    print_performance_timeline("bench.task.gpqa.gpqa_diamond", scores, "choice")


if __name__ == "__main__":
    main()
