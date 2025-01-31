
This repository contains the Python client library of [Epoch AI](https://epoch.ai/). At the moment, only one feature is supported: reading from our database of ML models and benchmark results.

## Installation

```bash
pip install epochai
```

## Usage
### Reading from our Airtable database of ML models and benchmark results
1. Open our [Airtable base](https://airtable.com/appsyxA7qAp1bvsrl/tblyjKGBmFS5khLdW/viwvuE5MiSv6wcyeW?blocks=hide)
2. Airtable doesn't allow public API access, so you'll have to make a copy of the base.
3. Define the `AIRTABLE_BASE_ID` environment variable with the ID of the base you just copied.
3. Create an Airtable API key with access to the base, and the following scopes: `data.records:read`, `schema.bases:read`. Define the `AIRTABLE_API_KEY` environment variable with the key.

Now, you can get started with our example script:
