# import pandas as pd

# # Load the CSV file
# df = pd.read_csv("filtered_df - Copy.csv")

# # Append '@gmail.com' to each value in the 'email' column
# df["email"] = df["email"].astype(str) + "@gmail.com"

# # Save the modified file
# df.to_csv("filtered_df_copy_updated.csv", index=False)

# print("Email column updated successfully!")


# import pandas as pd
# import sqlite3
# import bcrypt
# from datetime import datetime

# # Load CSV
# df = pd.read_csv("filtered_df_copy_updated.csv")

# # Keep only unique users
# unique_users = df[['user_id', 'Name', 'email', 'password']].drop_duplicates()

# # Connect to SQLite DB
# conn = sqlite3.connect("meals.db")
# cursor = conn.cursor()

# # Create users table
# cursor.execute("""
# CREATE TABLE IF NOT EXISTS users (
#     user_id INTEGER PRIMARY KEY,
#     email TEXT UNIQUE NOT NULL,
#     password_hash TEXT NOT NULL,
#     name TEXT,
#     created_at TEXT
# )
# """)

# # Insert each user
# for _, row in unique_users.iterrows():
#     user_id = int(row["user_id"])
#     email = row["email"]
#     name = row["Name"]
#     password = row["password"]

#     # Hash the password
#     password_hash = bcrypt.hashpw(str(password).encode(), bcrypt.gensalt()).decode()

#     # Insert
#     try:
#         cursor.execute("""
#             INSERT INTO users (user_id, email, password_hash, name, created_at)
#             VALUES (?, ?, ?, ?, ?)
#         """, (user_id, email, password_hash, name, datetime.now().strftime("%Y-%m-%d")))
#     except sqlite3.IntegrityError:
#         print(f"⚠️ User with email {email} already exists. Skipping.")

# conn.commit()
# conn.close()
# print("✅ Users table created and populated.")

import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('meals.db')
cursor = conn.cursor()

# Update the incorrect email addresses
cursor.execute("UPDATE users SET email = 'john@gmail.com' WHERE email = 'john@gmail.com@gmail.com'")
cursor.execute("UPDATE users SET email = '240@gmail.com' WHERE email = '240@gmail.com@gmail.com'")
cursor.execute("UPDATE users SET email = '338@gmail.com' WHERE email = '338@gmail.com@gmail.com'")

# Commit the changes and close the connection
conn.commit()
conn.close()
