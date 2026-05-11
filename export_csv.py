import sqlite3
import pandas as pd

# Connect database
conn = sqlite3.connect("retail_ai.db")

# Read analytics table
df = pd.read_sql_query(

    "SELECT * FROM analytics",

    conn

)

# Export CSV
df.to_csv(

    "retail_analytics_report.csv",

    index=False

)

conn.close()

print("CSV Report Exported Successfully!")