from flask import Flask, render_template, request, g, send_from_directory
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timedelta
import os

app = Flask(__name__)

@app.context_processor
def inject_current_year():
    return {'current_year': datetime.utcnow().year}

@app.route('/robots.txt')
def serve_robots():
    return send_from_directory('static', 'robots.txt')

def get_mongo_client():
    if 'mongo_client' not in g:
        uri = os.getenv("MONGO_URI")
        if not uri:
            raise ValueError("MONGO_URI environment variable must be set")
        g.mongo_client = MongoClient(uri)
    return g.mongo_client

@app.after_request
def add_header(response):
    response.cache_control.max_age = 600
    return response

@app.route('/')
def display_news():
    sort = request.args.get('sort')
    if not sort:
        sort = 'time'
    if sort != 'time' and sort != 'importance':
        sort = 'time'
    mongo_db = get_mongo_client()["nb3000"]
    stories_collection = mongo_db["stories"]

    # Fetch and sort stories by importance in descending order
    horizon = datetime.now() - timedelta(days=1)
    cursor = stories_collection.find({ "updated": {"$gt": horizon } })
    if sort == 'time':
        cursor = cursor.sort('updated', -1)
    elif sort == 'importance':
        cursor = cursor.sort([
            ('summary.importance', -1),
            ('updated', -1)  # Secondary sort by time descending
        ])
    stories = list(cursor)

    last_update_time = max(stories, key=lambda story: story['run_start_time'])['run_start_time']

    # Format stories for rendering
    formatted_stories = [
        {
            "_id": story.get("_id"),
            "headline": story.get("headline"),
            "alt_headline": story.get("summary", {}).get("title"),
            "updated": story.get("updated").strftime("%Y-%m-%d %H:%M UTC"),
            "link": story.get("link"),
            "summary": story.get("summary", {}).get("summary"),
            "importance": "\U0001F525" * story.get("summary", {}).get("importance", 0),
            "keywords": story.get("summary", {}).get("keywords"),
            "category": story.get("summary", {}).get("category"),
            "source": story.get("source"),
            "request": request
        }
        for story in stories
    ]

    return render_template("news.html",
        stories=formatted_stories,
        sort_by=sort,
        location="main",
        update_time=last_update_time.strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/story/<story_id>')
def display_story(story_id):
    mongo_db = get_mongo_client()["nb3000"]
    stories_collection = mongo_db["stories"]
    story = stories_collection.find_one({"_id": ObjectId(story_id)})
    
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
                'updated': -1
            }
        }
    ]
    similar_stories = list(stories_collection.aggregate(pipeline))
    
    similar_stories = [s for s in similar_stories if s['_id'] != story['_id']]
    
    for s in similar_stories:
        s['updated'] = s['updated'].strftime("%Y-%m-%d %H:%M UTC")

    return render_template("story.html", 
                           story=story, 
                           similar_stories=similar_stories,
                           importance="\U0001F525" * story.get("summary", {}).get("importance", 0))

@app.route('/category/<category>', defaults={'subcategory': None})
@app.route('/category/<category>/<subcategory>')
def display_category(category, subcategory):
    sort = request.args.get('sort')
    if not sort:
        sort = 'time'
    if sort != 'time' and sort != 'importance':
        sort = 'time'
    horizon = datetime.now() - timedelta(days=14)
    mongo_db = get_mongo_client()["nb3000"]
    stories_collection = mongo_db["stories"]
    cat = category + '/' + subcategory if subcategory else category
    cursor = stories_collection.find({ 'updated': {'$gt': horizon }, 'summary.categories': cat })
    if sort == 'time':
        cursor = cursor.sort('updated', -1)
    elif sort == 'importance':
        cursor = cursor.sort([
            ('summary.importance', -1),
            ('updated', -1)  # Secondary sort by time descending
        ])
    stories = list(cursor)
    last_update_time = max(stories, key=lambda story: story['run_start_time'])['run_start_time']
    formatted_stories = [
        {
            "_id": story.get("_id"),
            "headline": story.get("headline"),
            "alt_headline": story.get("summary", {}).get("title"),
            "updated": story.get("updated").strftime("%Y-%m-%d %H:%M UTC"),
            "link": story.get("link"),
            "summary": story.get("summary", {}).get("summary"),
            "importance": "\U0001F525" * story.get("summary", {}).get("importance", 0),
            "keywords": story.get("summary", {}).get("keywords"),
            "category": story.get("summary", {}).get("category"),
            "source": story.get("source"),
            "request": request
        }
        for story in stories
    ]

    return render_template("news.html",
        stories=formatted_stories,
        sort_by=sort,
        location="category/" + cat,
        update_time=last_update_time.strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/keyword/<keyword>')
def display_keyword(keyword):
    sort = request.args.get('sort')
    if not sort:
        sort = 'time'
    if sort != 'time' and sort != 'importance':
        sort = 'time'
    mongo_db = get_mongo_client()["nb3000"]
    keywords_col = mongo_db["keywords"]
    stories_col = mongo_db['stories']
    k = keywords_col.find_one({"keyword": keyword})
    emb = k["embedding"]

    pipeline = [
        {
            '$vectorSearch': {
                'index': 'embed_search',
                'path': 'embedding',
                'queryVector': emb,
                'numCandidates': 100,
                'limit': 10
            }
        },
        {
            '$project': {
                '_id': 0,
                'keyword': 1,
                'score': {
                    '$meta': 'vectorSearchScore'
                }
            }
        },
        {
            '$match': {
                'score': { '$gte': 0.9 }
            }
        }
    ]
    results = list(keywords_col.aggregate(pipeline))
    keywords = [k["keyword"] for k in results]

    cursor = stories_col.find({'summary.keywords': {"$in": keywords}})
    if sort == 'time':
        cursor = cursor.sort('updated', -1)
    elif sort == 'importance':
        cursor = cursor.sort([
            ('summary.importance', -1),
            ('updated', -1)  # Secondary sort by time descending
        ])
    stories = list(cursor)
    last_update_time = max(stories, key=lambda story: story['run_start_time'])['run_start_time']
    formatted_stories = [
        {
            "_id": story.get("_id"),
            "headline": story.get("headline"),
            "alt_headline": story.get("summary", {}).get("title"),
            "updated": story.get("updated").strftime("%Y-%m-%d %H:%M UTC"),
            "link": story.get("link"),
            "summary": story.get("summary", {}).get("summary"),
            "importance": "\U0001F525" * story.get("summary", {}).get("importance", 0),
            "keywords": story.get("summary", {}).get("keywords"),
            "category": story.get("summary", {}).get("category"),
            "source": story.get("source"),
            "request": request
        }
        for story in stories
    ]

    return render_template("news.html",
        stories=formatted_stories,
        sort_by=sort,
        location="keyword/" + keyword,
        update_time=last_update_time.strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    app.run(debug=True)