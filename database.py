import sqlite3
import pickle

DB_NAME = "textile.db"

def get_db_connection():
    con = sqlite3.connect(DB_NAME)
    con.row_factory = sqlite3.Row
    return con

def create_table():
    con = get_db_connection()
    try:
        con.execute('''
        CREATE TABLE IF NOT EXISTS textiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            features BLOB NOT NULL
        )
        ''')
        con.commit()
        print("Table created successfully!")
    except sqlite3.OperationalError as e:
        print(f"Error creating table: {e}")
    finally:
        con.close()
    
    
def insert_features(path, features):
    con = get_db_connection()
    cursor = con.cursor()
    
    serialized_features = pickle.dumps(features)
    
    cursor.execute('''
                   INSERT INTO textiles (path, features)
                   VALUES (?, ?)''',
                   (path, serialized_features))
    con.commit()
    con.close()

def get_features():
    con = get_db_connection()
    cursor = con.cursor()
    cursor.execute('SELECT * FROM textiles')
    rows = cursor.fetchall()
    con.close()
    
    json_data = []
    for row in rows:
        path = row['path']
        features = pickle.loads(row['features'])  
        json_data.append({
            'path': path,
            'features': features
        })
    return json_data

create_table()