from datetime import datetime, timedelta
from pymongo.mongo_client import MongoClient
import os

def main():
    mongo_uri = os.getenv("MONGO_URI")
    client = MongoClient(mongo_uri)
    db = client.get_database('nb3000')
    stories_col = db.get_collection('stories')
    
    stories = stories_col.find({"updated": {"$gt": datetime.now() + timedelta(days=1)}})
    for story in stories:
        stories_col.update_one({"_id": story["_id"]}, {"$set": {"updated": datetime.now()}})
        print(story["source"], story["headline"])

if __name__ == "__main__":
    main()