from backend.app.db.session import engine
from backend.app.db.models import Base

def reset_database():
    """Reset the database - WARNING: This deletes all data"""
    print("Dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    print("Creating all tables...")
    Base.metadata.create_all(bind=engine)
    print("Database reset complete!")

if __name__ == "__main__":
    reset_database()