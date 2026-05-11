import sqlite3

conn = sqlite3.connect("retail_ai.db")

cursor = conn.cursor()

# Visitor table
cursor.execute("""

CREATE TABLE IF NOT EXISTS visitors(

    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER

)

""")

# Analytics table
cursor.execute("""

CREATE TABLE IF NOT EXISTS analytics(

    id INTEGER PRIMARY KEY AUTOINCREMENT,
    people_count INTEGER,
    entry_count INTEGER,
    exit_count INTEGER,
    inside_count INTEGER

)

""")

conn.commit()
conn.close()

print("Database Created Successfully")