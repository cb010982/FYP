import sqlite3

def create_sugar_table():
    conn = sqlite3.connect("meals.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sugar_levels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        sugar_value REAL NOT NULL,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    """)

    conn.commit()
    conn.close()

create_sugar_table()
