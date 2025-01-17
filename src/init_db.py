from database import engine
from db_model import Base

def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()