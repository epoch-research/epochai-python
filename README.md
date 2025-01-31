This repository contains the Python client library of [Epoch AI](https://epoch.ai/). At the moment, only one feature is supported: reading from our database of ML models and benchmark results.

## Installation

```bash
pip install epochai
```

## Usage
### Reading from our Airtable database of ML models and benchmark results
A few preparatory steps are required: 
3. Open our [Airtable base](https://airtable.com/appsyxA7qAp1bvsrl/tblyjKGBmFS5khLdW/viwvuE5MiSv6wcyeW?blocks=hide)
2. Airtable doesn't allow public API access, so you'll have to make a copy of the base.
3. Define the `AIRTABLE_BASE_ID` environment variable with the ID of the base you just copied. (The ID is in the URL and starts with `app`.)
3. Create an Airtable API key with access to the base, and the following scopes: `data.records:read`, `schema.bases:read`. Define the `AIRTABLE_API_KEY` environment variable with the key.

You're now ready to use the library. The database models are defined in `epochai.airtable.models`. 

You can get started with our example script [`examples/airtable.py`](examples/airtable.py), or try the snippets below.

```python
from epochai.airtable.models import MLModel, Task, Score, Organization, BenchmarkRun

# Get everything at the start to minimize API calls
scores = Score.all(memoize=True)
runs = BenchmarkRun.all(memoize=True)
models = MLModel.all(memoize=True)
tasks = Task.all(memoize=True)
organizations = Organization.all(memoize=True)
```

Print information about a model:

```python
print_model_info("claude-3-5-sonnet-20240620")
```

<img src="assets/model.png" width="500"/>

Print the highest scores for a benchmark and scorer:

```python
print_high_scores(
    task_path="bench.task.hendrycks_math.hendrycks_math_lvl_5",
    scorer="model_graded_equiv",
    scores=scores
)
```

<img src="assets/highscores.png" width="500"/>

Track the best-performing model to date over time:
```python
print_performance_timeline(
    task_path="bench.task.gpqa.gpqa_diamond",
    scorer="choice",
    scores=scores
)
```

<img src="assets/timeline.png" width="500"/>