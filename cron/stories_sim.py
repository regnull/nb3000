from bson import ObjectId
from dotenv import load_dotenv
import os
from pymongo import MongoClient
from pprint import pprint

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, '../.env')
    print(f"Loading environment variables from {env_path}")
    load_dotenv(env_path)

    mongo_uri = os.getenv("MONGO_URI")
    client = MongoClient(mongo_uri)
    db = client.get_database('nb3000')
    stories_col = db.get_collection('stories')
    story = stories_col.find_one({"_id": ObjectId('68306b7d96735d46299476c5')})
    
    pipeline = [
        {
            '$vectorSearch': {
                'index': 'story_embed',
                'path': 'embedding',
                'queryVector': story['embedding'],
                'numCandidates': 100,
                'limit': 10
            }
        },
        {
            '$project': {
                '_id': 1,
                'summary': 1,
                'source': 1,
                'updated': 1,
                'score': {
                    '$meta': 'vectorSearchScore'
                }
            }
        },
        {
            '$match': {
                'score': { '$gte': 0.9 }
            }
        },
        {
            '$sort': {
                'score': -1
            }
        }
    ]
    similar_stories = list(stories_col.aggregate(pipeline))
    
    similar_stories = [s for s in similar_stories if s['_id'] != story['_id']]
    
    pprint(similar_stories)

if __name__ == "__main__":
    main()