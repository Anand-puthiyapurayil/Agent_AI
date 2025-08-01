from sqlalchemy import create_engine, text
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

engine = create_engine(os.getenv('DATABASE_URL'))

with engine.connect() as conn:
    result = conn.execute(text("SELECT * FROM deliveries LIMIT 5;"))
    df = pd.DataFrame(result.fetchall(), columns=result.keys())
    print(df.head())

with engine.connect() as conn:
    result = conn.execute(text("SELECT * FROM matches LIMIT 5;"))
    df = pd.DataFrame(result.fetchall(), columns=result.keys())
    print(df.head())

