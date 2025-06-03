import sqlite3
import pandas as pd

# === CONFIGURATION ===
DB_NAME = "meals.db"
CSV_PATH = "course_cleaned.csv"  # Ensure this file exists in the same folder

# === CONNECT TO DB ===
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

# === CREATE TABLE ===
cursor.execute('''
CREATE TABLE IF NOT EXISTS course_cleaned (
    course_id INTEGER PRIMARY KEY,
    course_name TEXT,
    review_nums INTEGER,
    category TEXT,  
    aver_rate REAL,
    image_url TEXT,
    ingredients TEXT,
    cooking_directions TEXT,
    reviews TEXT,
    tags TEXT,
    calories REAL,
    sugar REAL,
    fiber REAL
)
''')

conn.commit()
print("✅ Table 'course_cleaned' created successfully.")

# === LOAD AND INSERT CSV DATA ===
try:
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]

    df.to_sql("course_cleaned", conn, if_exists="append", index=False)
    print(f"✅ Inserted {len(df)} records into 'course_cleaned'.")
except Exception as e:
    print("❌ Error loading CSV:", e)

conn.close()



# import sqlite3
# import pandas as pd

# # === CONFIGURATION ===
# DB_NAME = "meals.db"

# # === CONNECT TO DB ===
# conn = sqlite3.connect(DB_NAME)
# cursor = conn.cursor()

# # === DROP existing course_cleaned table (if it exists) ===
# cursor.execute("DROP TABLE IF EXISTS meals")
# conn.commit()
# print("🗑️ Dropped existing 'meals' table.")