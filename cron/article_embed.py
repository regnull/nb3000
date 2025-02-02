from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient
import os

def main():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        dimensions=512
    )

    client = MongoClient(os.getenv("MONGO_URI"))
    db = client.get_database('nb3000')
    stories_col = db.get_collection('stories')

    stories = stories_col.find({})
    for story in stories:
        if story.get('embedding'):
            print(f"Skipping {story['headline']} because it already has an embedding")
            continue
        print(story['headline'])
        text = story['headline'] + "\n\n" + story['summary']['summary']
        embedding_vector = embeddings.embed_query(text)
        story['embedding'] = embedding_vector
        stories_col.update_one({'_id': story['_id']}, {'$set': {'embedding': embedding_vector}})

if __name__ == "__main__":
    main()
