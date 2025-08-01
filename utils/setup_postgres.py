import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)

# Load CSV datasets (adjust file names if needed)
df1 = pd.read_csv('data/deliveries.csv')
df2 = pd.read_csv('data/matches.csv')

# Check columns (to confirm schema)
print("deliveries Columns:", df1.columns)
print("matches Columns:", df2.columns) 


# Insert into PostgreSQL tables
df1.to_sql('deliveries', engine, if_exists='replace', index=False)
df2.to_sql('matches', engine, if_exists='replace', index=False)

print("✅ Data successfully loaded into PostgreSQL tables: deliveries, matches.")