from pyairtable.orm import Model, fields as F
from environs import Env
from epochai.airtable.client import AIRTABLE_TOKEN, BASE_ID
env = Env()
env.read_env()


def create_meta(table_name: str):
    return {
        "api_key": AIRTABLE_TOKEN,
        "base_id": BASE_ID,
        "table_name": table_name,
    }


class BenchmarkRun(Model):
    task = F.SingleLinkField("task", "Task", raise_if_many=True)  # type: ignore
    model = F.SingleLinkField("model", "MLModel", raise_if_many=True)  # type: ignore
    scores = F.LinkField("Scores", "Score")  # type: ignore

    logs = F.UrlField("logs")  # type: ignore
    log_viewer = F.TextField("log viewer")  # type: ignore
    job = F.UrlField("job")  # type: ignore

    inspect_run_id = F.TextField("id")  # type: ignore
    started_at = F.DatetimeField("started_at")  # type: ignore
    succeeded_at = F.DatetimeField("succeeded_at")  # type: ignore

    Meta = create_meta(table_name="benchmarks/runs")


class Task(Model):
    path = F.TextField("path")
    benchmark_runs = F.LinkField("BenchmarkRuns", BenchmarkRun)

    Meta = create_meta(table_name="benchmarks/tasks")


class MLModel(Model):
    model_id = F.TextField("id")  # type: ignore
    model_group = F.SingleLinkField("Model", "MLModelGroup", raise_if_many=True)  # type: ignore
    benchmark_runs = F.LinkField("benchmarks/runs", BenchmarkRun)
    hf_developer = F.TextField("Hugging Face developer id")  # type: ignore
    release_date = F.DatetimeField("Version release date")  # type: ignore

    Meta = create_meta(table_name="Model versions")


class MLModelGroup(Model):
    organizations = F.LinkField("Organization", "Organization")  # type: ignore
    Meta = create_meta(table_name="ML Models")


class Organization(Model):
    name = F.TextField("Organization")
    Meta = create_meta(table_name="Organizations")


class Score(Model):
    scorer = F.TextField("scorer")
    mean = F.FloatField("mean")
    stderr = F.FloatField("stderr")
    benchmark_run = F.SingleLinkField("BenchmarkRuns", BenchmarkRun)

    Meta = create_meta(table_name="benchmarks/scores")
