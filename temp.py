from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# SQLite database path
DATABASE_URL = "sqlite:///data/social_media_data.db"


# Create engine
engine = create_engine(
    DATABASE_URL, echo=True, connect_args={"check_same_thread": False}
)


# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


Base = declarative_base()


# Check database tables
def get_tables():

    inspector = inspect(engine)

    tables = inspector.get_table_names()

    return tables


# Retrieve all records from table
def fetch_all(table_name):

    with engine.connect() as connection:

        result = connection.execute(f"SELECT * FROM {table_name}")

        rows = result.fetchall()

        return rows


if __name__ == "__main__":

    print("Available Tables:")

    tables = get_tables()

    for table in tables:
        print(table)

    if tables:

        data = fetch_all(tables[0])

        print("\nData:")

        for row in data:
            print(row)
