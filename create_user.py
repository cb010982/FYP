# import sqlite3
# import bcrypt
# from datetime import datetime

# def create_user(email, password, name):
#     conn = sqlite3.connect("meals.db")
#     cursor = conn.cursor()

#     hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

#     try:
#         cursor.execute("""
#             INSERT INTO users (email, password_hash, name, created_at)
#             VALUES (?, ?, ?, ?)
#         """, (email, hashed, name, datetime.now().strftime("%Y-%m-%d")))
#         conn.commit()
#         return "User created successfully"
#     except sqlite3.IntegrityError:
#         return "Email already exists"
#     finally:
#         conn.close()



# import sqlite3
# conn = sqlite3.connect("meals.db")
# cursor = conn.cursor()
# cursor.execute("""
#     CREATE TABLE IF NOT EXISTS rl_buffer (
#         user_index INTEGER,
#         course_index INTEGER,
#         rating INTEGER
#     )
# """)
# conn.commit()
# conn.close()

import sqlite3
conn = sqlite3.connect("meals.db")
cursor = conn.cursor()
cursor.execute("DELETE FROM rl_buffer")
conn.commit()
conn.close()
