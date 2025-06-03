import sqlite3
import pandas as pd

# # === CONFIGURATION ===
# DB_NAME = "meals.db"
# CSV_PATH = "filtered_df.csv"  # Make sure this file exists in the same folder

# # === CONNECT TO DB ===
# conn = sqlite3.connect(DB_NAME)
# cursor = conn.cursor()

# # === CREATE TABLE ===
# cursor.execute('''
# CREATE TABLE IF NOT EXISTS filtered_df (
#     user_id INTEGER,
#     name TEXT,
#     course_id INTEGER,
#     rating INTEGER CHECK(rating IN (0,1)),
#     course_name TEXT,
#     category INTEGER,
#     image_url TEXT,
#     ingredients TEXT,
#     cooking_directions TEXT,
#     calories REAL,
#     sugar REAL,
#     fiber REAL
# )
# ''')

# conn.commit()
# print("✅ Table 'filtered_df' created successfully.")

# # === LOAD AND INSERT CSV DATA ===
# try:
#     df = pd.read_csv(CSV_PATH)
#     # Optional: clean column names to match DB
#     df.columns = [c.strip() for c in df.columns]
#     df.to_sql("filtered_df", conn, if_exists="append", index=False)
#     print(f"✅ Inserted {len(df)} records into 'filtered_df'.")
# except Exception as e:
#     print("❌ Error loading CSV:", e)

# conn.close()


# conn = sqlite3.connect("meals.db")
# cursor = conn.cursor()
# cursor.execute("ALTER TABLE filtered_df ADD COLUMN user_index INTEGER")
# conn.commit()
# conn.close()

import pandas as pd

# Load the CSV
df = pd.read_csv("filtered_df.csv")

# Define expected columns in the correct order
expected_cols = [
    "user_id", "name", "email", "password",
    "course_id", "rating", "course_name", "category",
    "image_url", "ingredients", "cooking_directions",
    "calories", "sugar", "fiber", "user_index"
]

# Add missing columns (if any) with NaNs
for col in expected_cols:
    if col not in df.columns:
        df[col] = None

# Reorder columns
df = df[expected_cols]

# Overwrite the CSV with fixed structure
df.to_csv("filtered_df.csv", index=False)

print("✅ filtered_df.csv column order fixed!")




# import sqlite3
# import pandas as pd

# # Connect to the database
# conn = sqlite3.connect("meals.db")
# cursor = conn.cursor()

# # Load the table
# df = pd.read_sql("SELECT * FROM filtered_df", conn)

# # Drop the last 27 rows
# df_trimmed = df[:-27]

# # Replace the table
# df_trimmed.to_sql("filtered_df", conn, if_exists="replace", index=False)

# conn.close()
# print("✅ Deleted last 27 rows from 'filtered_df' table in meals.db")





