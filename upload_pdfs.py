import os
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from gridfs import GridFS
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI not found in .env file")

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    db = client["university_db"]
    fs = GridFS(db)
    print("Connected to MongoDB Atlas")

    pdfs = [
        {"university_id": "24-25CA.WUCatalog", "path": "data/24-25CA.WUCatalog.pdf"},
        {"university_id": "App Form MBA 2026", "path": "data/App Form MBA 2026.pdf"}
    ]

    for pdf in pdfs:
        if not os.path.exists(pdf["path"]):
            print(f"PDF not found: {pdf['path']}")
            continue
        with open(pdf["path"], 'rb') as f:
            fs.put(f, filename=pdf["university_id"] + ".pdf", metadata={"university_id": pdf["university_id"]})
        print(f"Uploaded {pdf['university_id']}.pdf to GridFS")
except ServerSelectionTimeoutError as e:
    print(f"Failed to connect to MongoDB: {str(e)}")
except Exception as e:
    print(f"Error: {str(e)}")
finally:
    client.close()