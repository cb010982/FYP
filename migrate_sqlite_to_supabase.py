# import sqlite3
# import psycopg2
# import pandas as pd

# # === 1. Connect to SQLite ===
# sqlite_conn = sqlite3.connect("meals.db")

# # Tables to migrate
# tables = ["course_cleaned", "rl_buffer", "users", "sugar_levels", "filtered_df"]

# # === 2. Connect to Supabase PostgreSQL ===
# pg_conn = psycopg2.connect(
#     host="db.dzlxqbivymultdwzqljn.supabase.co",
#     port="5432",
#     database="postgres",
#     user="postgres",
#     password="Senu@123$"
# )
# pg_cursor = pg_conn.cursor()

# # === 3. Function to insert data ===
# def migrate_table(table_name):
#     df = pd.read_sql_query(f"SELECT * FROM {table_name}", sqlite_conn)

#     print(f"Migrating {table_name} ({len(df)} rows)...")

#     # Build insert query dynamically
#     for index, row in df.iterrows():
#         columns = ', '.join(df.columns)
#         placeholders = ', '.join(['%s'] * len(row))
#         values = tuple(row)
#         sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

#         try:
#             pg_cursor.execute(sql, values)
#         except Exception as e:
#             pg_conn.rollback()  # resets transaction so we can continue
#             print(f" Error in row {index}: {row.to_dict()}")
#             print(f" Postgres error: {e}")

    
#     pg_conn.commit()
#     print(f"{table_name} migration done.\n")

# # === 4. Migrate All Tables ===
# for table in tables:
#     migrate_table(table)

# # === 5. Close Connections ===
# pg_cursor.close()
# pg_conn.close()
# sqlite_conn.close()
# print(" All done.")
import sqlite3
import psycopg2
import pandas as pd

# === 1. Connect to SQLite ===
sqlite_conn = sqlite3.connect("meals.db")

# === 2. Connect to Supabase PostgreSQL ===
pg_conn = psycopg2.connect(
    host="aws-0-ap-southeast-1.pooler.supabase.com",
    port="5432",
    database="postgres",
    user="postgres.stkutmsbmwxrbnebxmoh",
    password="Senu@123$"
)

pg_cursor = pg_conn.cursor()

# === 3. Function to insert data ===
def migrate_table(table_name):
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", sqlite_conn)

    print(f"Migrating {table_name} ({len(df)} rows)...")

    # Build insert query dynamically
    for index, row in df.iterrows():
        columns = ', '.join(df.columns)
        placeholders = ', '.join(['%s'] * len(row))
        values = tuple(row)
        sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

        try:
            pg_cursor.execute(sql, values)
        except Exception as e:
            pg_conn.rollback()
            print(f" Error in row {index}: {row.to_dict()}")
            print(f" Postgres error: {e}")

    pg_conn.commit()
    print(f"{table_name} migration done.\n")

# === 4. Migrate Only course_cleaned ===
migrate_table("course_cleaned")

# === 5. Close Connections ===
pg_cursor.close()
pg_conn.close()
sqlite_conn.close()
print(" All done.")
