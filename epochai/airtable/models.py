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

    # The corresponding files can be read using Inspect’s `read_eval_log` function, documented here:
    # https://inspect.ai-safety-institute.org.uk/reference/inspect_ai.log.html#read_eval_log
    logs = F.UrlField("logs")  # type: ignore
    log_viewer = F.TextField("log viewer")  # type: ignore
    job = F.UrlField("job")  # type: ignore

    inspect_run_id = F.TextField("id")  # type: ignore
    started_at = F.DatetimeField("started_at")  # type: ignore
    status = F.SelectField("Status")  # type: ignore

    Meta = create_meta(table_name="benchmarks/runs")

    def read_eval_log(self, **kwargs):
        """
        Read this run's eval log using Inspect's `read_eval_log` function.

        The log is fetched from the URL in the `logs` field (typically an S3
        path, which requires the relevant credentials to be configured).
        Extra keyword arguments are passed through to `read_eval_log`.

        Requires the `inspect-ai` package to be installed.
        """
        try:
            from inspect_ai.log import read_eval_log
        except ImportError as e:
            raise ImportError(
                "Reading eval logs requires the `inspect-ai` package: pip install inspect-ai"
            ) from e

        return read_eval_log(self.logs, **kwargs)

    def response_lengths(self) -> list[int]:
        """
        Response length, in output tokens, for each sample in this run.

        The lengths are computed from the run's eval log (see `read_eval_log`),
        so no manual log handling is required. Samples without recorded token
        usage (e.g. failed samples) count as 0.
        """
        log = self.read_eval_log()
        lengths = []
        for sample in log.samples or []:
            usage = sample.output.usage
            if usage and usage.output_tokens is not None:
                lengths.append(usage.output_tokens)
            else:
                lengths.append(0)
        return lengths


class Task(Model):
    path = F.TextField("path")
    name = F.TextField("Name")
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

    accessibility = F.TextField("Model accessibility")
    training_compute = F.FloatField("Training compute (FLOP)")


class Organization(Model):
    name = F.TextField("Organization")
    Meta = create_meta(table_name="Organizations")


class Score(Model):
    scorer = F.TextField("scorer")
    mean = F.FloatField("mean")
    stderr = F.FloatField("stderr")
    benchmark_run = F.SingleLinkField("BenchmarkRuns", BenchmarkRun)

    Meta = create_meta(table_name="benchmarks/scores")
