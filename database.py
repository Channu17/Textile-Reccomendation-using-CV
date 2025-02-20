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
            link TEXT UNIQUE NOT NULL,
            features BLOB NOT NULL
        )
        ''')
        con.commit()
        print("Table created successfully!")
    except sqlite3.OperationalError as e:
        print(f"Error creating table: {e}")
    finally:
        con.close()
    
    
def insert_features(link, features):
    con = get_db_connection()
    cursor = con.cursor()
    
    serialized_features = pickle.dumps(features)
    
    cursor.execute('''
                   INSERT INTO textiles (link, features)
                   VALUES (?, ?)''',
                   (link, serialized_features))
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
        link = row['link']
        features = pickle.loads(row['features']).reshape(-1)
        json_data.append({
            'link': link,
            'features': features
        })
    return json_data

create_table()