import os

# The models module reads these env vars at import time; tests never hit the network.
os.environ.setdefault("AIRTABLE_PERSONAL_ACCESS_TOKEN", "test-token")
os.environ.setdefault("AIRTABLE_BASE_ID", "appTEST")
