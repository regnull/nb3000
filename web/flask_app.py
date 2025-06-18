from flask import Flask, render_template, request, g, send_from_directory, abort
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timedelta
import os
import dateparser
from ip_blocker import IPBlocker

app = Flask(__name__)

# Initialize IP blocker
ip_blocker = IPBlocker()

bots = [
    'SemrushBot',
    'AhrefsBot',
    'Bingbot',
    'YandexBot',
]

@app.context_processor
def inject_current_year():
    return {'current_year': datetime.utcnow().year}

@app.route('/robots.txt')
def serve_robots():
    return send_from_directory('static', 'robots.txt')

@app.route('/ads.txt')
def serve_ads():
    return send_from_directory('static', 'ads.txt')


def get_mongo_client():
    if 'mongo_client' not in g:
        uri = os.getenv("MONGO_URI")
        if not uri:
            raise ValueError("MONGO_URI environment variable must be set")
        g.mongo_client = MongoClient(uri)
    return g.mongo_client

@app.before_request
def block_bots():
    # Get user agent and convert to lowercase for case-insensitive matching
    user_agent = request.headers.get('User-Agent', '').lower()
    print(f"DEBUG: Received request with User-Agent: {user_agent}")
    
    # Check if user agent contains any known bot identifiers
    for bot in bots:
        if bot.lower() in user_agent:
            print(f"DEBUG: Blocking bot: {bot} found in {user_agent}")
            abort(403)  # Forbidden
    
    # Check IP blocking
    client_ip = request.remote_addr
    print(f"DEBUG: Checking IP: {client_ip}")
    if ip_blocker.is_blocked(client_ip):
        print(f"DEBUG: Blocking IP: {client_ip}")
        abort(403)  # Forbidden
    else:
        print(f"DEBUG: IP {client_ip} is not blocked")

@app.after_request
def add_header(response):
    response.cache_control.max_age = 600
    return response

@app.route('/stories')
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

    # Handle case where there might be no stories after filtering
    if not stories:
        last_update_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_stories = []
    else:
        last_update_time = max(stories, key=lambda story: story['run_start_time'])['run_start_time']
        last_update_time_str = last_update_time.strftime("%Y-%m-%d %H:%M:%S")
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
        location="stories",
        update_time=last_update_time_str)

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
    cursor = cursor.allow_disk_use(True)
    stories = list(cursor)

    if not stories:
        last_update_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_stories = []
    else:
        last_update_time = max(stories, key=lambda story: story['run_start_time'])['run_start_time']
        last_update_time_str = last_update_time.strftime("%Y-%m-%d %H:%M:%S")
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
        update_time=last_update_time_str)

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
    
    # Check if the keyword was found
    if k is None:
        # Keyword not found, return empty results or a specific message
        # For now, let's return an empty list of stories and a current time update
        return render_template("news.html",
                               stories=[],
                               sort_by=sort,
                               location="keyword/" + keyword,
                               update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                               error_message=f"Keyword '{keyword}' not found.")
    
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
    cursor = cursor.allow_disk_use(True)
    stories = list(cursor)

    if not stories:
        last_update_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_stories = []
    else:
        last_update_time = max(stories, key=lambda story: story['run_start_time'])['run_start_time']
        last_update_time_str = last_update_time.strftime("%Y-%m-%d %H:%M:%S")
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
        update_time=last_update_time_str)

@app.route('/') # Changed from /topics to /
def display_topics():
    mongo_db = get_mongo_client()["nb3000"]
    topics_collection = mongo_db["topics"]
    stories_collection = mongo_db["stories"]
    news_summaries_collection = mongo_db["news_summaries"] # Added news_summaries collection

    # Fetch the latest daily news summary
    latest_daily_summary = news_summaries_collection.find_one(sort=[('date', -1)])

    # Define the time horizon for fetching topics (last 48 hours)
    horizon = datetime.now() - timedelta(hours=48)

    # Fetch topics sorted by last updated time, filtered by the horizon
    topics_cursor = topics_collection.find(
        {'updated': {'$gte': horizon}}
    ).sort('updated', -1).allow_disk_use(True) # -1 for descending
    
    processed_topics = []
    for topic in topics_cursor:
        # For each topic, fetch its articles
        article_ids = topic.get('stories', [])
        if not article_ids: # Skip if a topic somehow has no story IDs
            continue
        
        # Ensure article_ids are ObjectIds if they aren't already (they should be)
        object_id_article_ids = [ObjectId(id_val) for id_val in article_ids]
        
        articles_cursor = stories_collection.find({
            '_id': { '$in': object_id_article_ids }
        }).sort('updated', -1) # Sort articles within a topic by their update time
        
        articles_in_topic = list(articles_cursor)
        
        # Format articles for display (similar to other routes)
        formatted_articles = []
        for article in articles_in_topic:
            formatted_articles.append({
                "_id": article.get("_id"),
                "headline": article.get("headline"),
                "alt_headline": article.get("summary", {}).get("title"),
                "updated": article.get("updated").strftime("%Y-%m-%d %H:%M UTC") if article.get("updated") else "N/A",
                "link": article.get("link"),
                "summary": article.get("summary", {}).get("summary"),
                "importance_score": article.get("summary", {}).get("importance", 0), # Keep numeric for template logic
                "importance": "\U0001F525" * article.get("summary", {}).get("importance", 0),
                "keywords": article.get("summary", {}).get("keywords"),
                "category": article.get("summary", {}).get("category"),
                "source": article.get("source")
            })

        # Topic summary data
        topic_summary_data = topic.get('summary', {})
        
        processed_topics.append({
            '_id': topic.get('_id'),
            'updated': topic.get('updated').strftime("%Y-%m-%d %H:%M UTC") if topic.get('updated') else "N/A",
            'source': topic.get('source'),
            'title': topic_summary_data.get('title', 'Topic Title Missing'),
            'summary_text': topic_summary_data.get('summary', 'Topic summary missing.'),
            'importance_score': topic_summary_data.get('importance', 0),
            'importance': "\U0001F525" * topic_summary_data.get('importance', 0),
            'keywords': topic_summary_data.get('keywords', []),
            'category': topic_summary_data.get('category', 'N/A'),
            'articles': formatted_articles,
            'article_count': len(formatted_articles)
        })
        
    # Attempt to get a global last update time if possible, or use current time
    # This might need refinement based on how 'run_start_time' is stored or if a global update marker exists
    last_update_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if processed_topics and processed_topics[0]['articles']:
        # Try to get from the latest article of the latest topic if available
        # This assumes articles have 'run_start_time', which they do in 'display_news'
        # However, formatted_articles above doesn't include it. For simplicity, using topic updated time.
        latest_topic_update_dt = dateparser.parse(processed_topics[0]['updated'])
        if latest_topic_update_dt:
            last_update_time_str = latest_topic_update_dt.strftime("%Y-%m-%d %H:%M:%S")

    return render_template("topics.html",
                           topics=processed_topics,
                           latest_daily_summary=latest_daily_summary, # Pass daily summary to template
                           location="main", # Changed from "topics" to "main"
                           update_time=last_update_time_str)

@app.route('/topic/<topic_id>')
def display_topic_detail(topic_id):
    mongo_db = get_mongo_client()["nb3000"]
    topics_collection = mongo_db["topics"]
    stories_collection = mongo_db["stories"]

    try:
        topic = topics_collection.find_one({"_id": ObjectId(topic_id)})
    except Exception as e:
        # Handle cases where topic_id might not be a valid ObjectId, though ObjectId() itself can raise InvalidId
        print(f"Error fetching topic by ID {topic_id}: {e}")
        return "Topic not found or invalid ID", 404

    if not topic:
        return "Topic not found", 404

    # Fetch associated articles for the topic
    article_ids = topic.get('stories', [])
    articles_in_topic = []
    if article_ids:
        object_id_article_ids = [ObjectId(id_val) for id_val in article_ids]
        articles_cursor = stories_collection.find({
            '_id': { '$in': object_id_article_ids }
        }).sort('updated', -1) # Sort articles within topic by their update time
        articles_in_topic = list(articles_cursor)

    # Format articles for detailed display in the template
    formatted_articles = []
    for article in articles_in_topic:
        formatted_articles.append({
            "_id": article.get("_id"),
            "headline": article.get("headline"),
            "alt_headline": article.get("summary", {}).get("title"), # If the story was summarized, this is its title
            "link": article.get("link"),
            "source": article.get("source"),
            "updated": article.get("updated").strftime("%Y-%m-%d %H:%M UTC") if article.get("updated") else "N/A",
            "summary_text": article.get("summary", {}).get("summary"), # The article's own summary
            "keywords": article.get("summary", {}).get("keywords"),
            "category": article.get("summary", {}).get("category"),
            "importance_score": article.get("summary", {}).get("importance", 0),
            "importance_icons": "\U0001F525" * article.get("summary", {}).get("importance", 0)
        })

    # Prepare topic data for the template
    topic_summary_data = topic.get('summary', {})
    processed_topic = {
        '_id': topic.get('_id'),
        'title': topic_summary_data.get('title', 'Topic Title Missing'),
        'summary_text': topic_summary_data.get('summary', 'Topic summary missing.'),
        'updated': topic.get('updated').strftime("%Y-%m-%d %H:%M UTC") if topic.get('updated') else "N/A",
        'source': topic.get('source'), # This is the topic's source, e.g., "multiple"
        'keywords': topic_summary_data.get('keywords', []),
        'category': topic_summary_data.get('category', 'N/A'),
        'importance_score': topic_summary_data.get('importance', 0),
        'importance_icons': "\U0001F525" * topic_summary_data.get('importance', 0),
        'articles': formatted_articles,
        'article_count': len(formatted_articles)
    }

    return render_template("topic_detail.html", 
                           topic=processed_topic,
                           location="topic_detail") # For potential nav highlighting or other logic

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)