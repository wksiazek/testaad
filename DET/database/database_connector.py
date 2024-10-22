import sqlite3


class BaseSQLiteConnector:
    def __init__(self, database):
        self.database = database
        self.connection = None
        self.cursor = None

    def connect(self):
        self.connection = sqlite3.connect(self.database)
        self.cursor = self.connection.cursor()

    def execute_query(self, query, params=None):
        if params:
            self.cursor.execute(query, params)
        else:
            self.cursor.execute(query)

    def executemany_query(self, query, params=None):
        if params:
            self.cursor.executemany(query, params)
        else:
            self.cursor.executemany(query)

    def fetch_all(self):
        return self.cursor.fetchall()

    def fetch_one(self):
        return self.cursor.fetchone()

    def commit(self):
        self.connection.commit()

    def close(self):
        self.cursor.close()
        self.connection.close()


class SQLiteConnector(BaseSQLiteConnector):
    def __init__(self, database):
        super().__init__(database)

    def create_table(self, table_name):
        new_table_name = self.find_valid_table_name(table_name)

        create_table_query = f'''
            CREATE TABLE {new_table_name} (
                id INTEGER PRIMARY KEY,
                epoch INTEGER,
                argumentsBest TEXT,
                fitnessValueBest REAL,
                argumentsWorst TEXT,
                fitnessValueWorst REAL,
                mean REAL,
                std REAL,
                calculationTime REAL,
                population TEXT
            )
        '''

        self.execute_query(create_table_query)
        self.commit()

        return new_table_name

    def find_valid_table_name(self, table_name):
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?"
        self.execute_query(query, (table_name + '%',))
        results = self.fetch_all()
        tables = [row[0] for row in results]

        return f"{table_name}_{len(tables) + 1}"

    def find_existing_table_name(self, table_name):
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?"
        self.execute_query(query, (table_name + '%',))
        results = self.fetch_all()
        tables = [row[0] for row in results]

        return f"{table_name}_{len(tables)}"

    def insert_multiple_best_individuals(self, table_name, params):
        insert_query = f'''
        INSERT INTO {table_name} (epoch, argumentsBest, fitnessValueBest, argumentsWorst, fitnessValueWorst, mean, std, calculationTime, population)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''

        self.executemany_query(insert_query, params)
        self.commit()

    def get_results(self, table_name):
        query = f'''
        SELECT * FROM {table_name}
        '''

        self.execute_query(query)
        return self.fetch_all()
