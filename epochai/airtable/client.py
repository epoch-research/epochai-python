from environs import Env
from pyairtable import Api

env = Env()
env.read_env()

AIRTABLE_TOKEN = env("AIRTABLE_PERSONAL_ACCESS_TOKEN")
BASE_ID = env("AIRTABLE_BASE_ID")

client = Api(AIRTABLE_TOKEN)
base = client.base(BASE_ID)
