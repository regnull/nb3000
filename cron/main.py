import re
import os
import pytz
import pprint
import requests
import dateparser
import html # Import html module for escaping
import json # Added json for stringifying chunks for the new LLM step

from bs4 import BeautifulSoup
from pymongo.mongo_client import MongoClient
from pymongo.collection import Collection
from bson import ObjectId
from datetime import datetime, timedelta
from llm import (
    summarize_article, 
    get_text_embeddings, 
    summarize_stories, 
    generate_simple_daily_summary, # Updated import for simplified approach
    generate_topic_short_name,          
    DailyNewsSummary # For constructing the final object for DB
)
from dotenv import load_dotenv
from csm import ChristianScienceMonitor
from npr import NPR
from apnews import AssociatedPress
from daily_summary_generator import create_and_save_daily_summary # New import

def merge_stories(stories: list[dict], story: dict):
    for s in stories:
        if s['_id'] == story['_id']:
            continue
        story['headline'] += "\n\n" + s['headline']
        story['summary'] += "\n\n" + s['summary']
    return story

def find_similar_stories(embedding: list[float], stories_col: Collection):
    pipeline = [
        {
            '$vectorSearch': {
                'index': 'story_embed',
                'path': 'embedding',
                'queryVector': embedding,
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
                'topic': 1,
                'headline': 1,
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
    # similar_stories = [s for s in similar_stories if s['_id'] != story['_id']]
    return similar_stories

def fetch_cnn_lite_content():
    # URL for CNN Lite
    url = "https://lite.cnn.com/"

    try:
        # Fetch the content of the page
        response = requests.get(url)
        response.raise_for_status()

        # Parse the page with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract headlines and links
        articles = []
        for link in soup.find_all('a', href=True):
            headline = link.get_text(strip=True)
            href = link['href']

            # CNN Lite links are often relative, so we create the full URL
            full_url = url + href.lstrip('/') if href.startswith("/") else href

            # Only append articles matching the CNN Lite DG pattern
            # Pattern: lite.cnn.com followed by section and -dg suffix
            pattern = r"https://lite\.cnn\.com/\d{4}/\d{2}/\d{2}/[a-z-]+/[a-z0-9-]+"
            if re.match(pattern, full_url):
                articles.append({
                    "headline": headline,
                    "link": full_url,
                    "source": "CNN"
                })
                
        return articles

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching CNN Lite content: {e}")
        return []

def fetch_url_text(url, parse_timestamp=True):
    try:
        # Fetch the content of the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        parsed_timestamp = None
        if parse_timestamp:
            timestamp_element = soup.find('p', class_='timestamp--lite')
            parsed_timestamp = None

            if timestamp_element:
                timestamp_text = timestamp_element.get_text(strip=True).replace("Updated: ", "")
                timestamp_text = timestamp_text.strip()
                parsed_timestamp = dateparser.parse(timestamp_text)

        # Extract and return only the text
        text = soup.get_text(strip=True)  # strip=True removes extra whitespace
        return text, parsed_timestamp
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}", None
    

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, '../.env')
    print(f"Loading environment variables from {env_path}")
    load_dotenv(env_path)

    run_start_time = datetime.now()

    # NPR
    npr = NPR()
    articles = npr.fetch_articles()

    # Christian Science Monitor
    csm = ChristianScienceMonitor()
    articles.extend(csm.fetch_articles())
    
    # Associated Press
    ap = AssociatedPress()
    articles.extend(ap.fetch_articles())

    # CNN Lite
    articles.extend(fetch_cnn_lite_content())

    mongo_uri = os.getenv("MONGO_URI")
    client = MongoClient(mongo_uri)
    db = client.get_database('nb3000')
    stories_col = db.get_collection('stories')
    topics_col = db.get_collection("topics") # Ensure topics_col is defined here
    keywords_col = db["keywords"] # Ensure keywords_col is defined here (though not directly used by daily summary fn)
    news_summaries_col = db.get_collection('news_summaries') # Define news_summaries_col
    
    print("Fetching and analyzing stories...")
    processed_articles = []
    for article in articles:
        print(f"Processing article {article['headline']}...")

        # If an article with the same headline exists, we skip.
        # At some point, we might want to update the article if the timestamp is different.
        if stories_col.find_one({ "headline": article["headline"] }):
            print("article exists, skipping")
            continue
        
        if stories_col.find_one({ "link": article["link"] }):
            print("article exists, skipping")
            continue
        
        text, timestamp = fetch_url_text(article['link'], parse_timestamp=(article["source"] == "CNN"))
        if timestamp:
            article['updated'] = timestamp

        summary = summarize_article(text)
        if 'Error' in summary:
            print(f"Error processing article {article['link']}: {summary}")
            continue
        
        if summary['language'].lower() != 'english':
            print(f"Article {article['headline']} is not in English, skipping")
            continue
        
        if summary.get('time') is not None:
            aware_dt = summary.get('time')
            if aware_dt > datetime.now(pytz.utc):
                print(f"Article {article['headline']} has a future timestamp: {summary['time']}")
                summary['time'] = datetime.now()
        
        embed_text = article['headline'] + "\n\n" + summary['summary']
        embedding = get_text_embeddings(embed_text, model='text-embedding-3-small', dimensions=512)
        article['embedding'] = embedding
        article['summary'] = summary
        article['run_start_time'] = run_start_time
        if article.get('updated') is None:
            if article['summary'].get('time') is not None:
                article['updated'] = article['summary']['time']
            else:
                article['updated'] = datetime.now()

        parts = article['summary']['category'].split('/')
        category = ''
        categories = []
        for p in parts:
            if len(category) > 0:
                category += '/'
            category += p
            categories.append(category)
        article['summary']['categories'] = categories
        processed_articles.append(article)

    if len(processed_articles) == 0:
        print("No new articles to add")
        exit()

    keywords_col = db["keywords"]
    for article in processed_articles:
        for keyword in article["summary"]["keywords"]:
            print(f'processing keyword: {keyword}')
            k = keywords_col.find_one({"keyword": keyword})
            if k is not None:
                print(f"Keyword {keyword} already exists")
                continue
            embedding = get_text_embeddings(keyword)
            keywords_col.insert_one({"keyword": keyword, "embedding": embedding})
 
    # Add stories and topics
    topics_col = db["topics"]
    for article in processed_articles:
        similar_stories = find_similar_stories(article['embedding'], stories_col)
        # Filter out stories that don't have a topic
        similar_stories = [s for s in similar_stories if s.get('topic') is not None]
        if len(similar_stories) > 0:
            topic_id = similar_stories[0]['topic']
            summary = summarize_stories(similar_stories + [article])
            # Make sure all the stories refer to the same topic
            for a in similar_stories:
                stories_col.update_one({ "_id": a['_id'] }, { "$set": { "topic": topic_id } })
            article['topic'] = topic_id
            article_id = stories_col.insert_one(article).inserted_id
            ids = list(map(lambda s: s['_id'], similar_stories))
            pprint.pprint(ids)
            ids += [article['_id']]
            pprint.pprint(ids)
            print(f"summarized topic: {topic_id}")
            pprint.pprint(summary)
            pprint.pprint(ids)
            topics_col.update_one({ "_id": topic_id }, { "$set": 
                { 
                    "updated": datetime.now(),
                    "source": "multiple",
                    "stories": ids,
                    "summary": summary
                } 
            })
        else:
            topic_id = topics_col.insert_one(article).inserted_id
            article['topic'] = topic_id
            article_id = stories_col.insert_one(article).inserted_id
            topics_col.update_one({ "_id": topic_id }, { "$push": { "stories": article_id } })
    
    print("\nAdded articles:")
    for article in processed_articles:
        print("\n" + "="*80)
        print(f"IMPORTANCE: {article['summary'].get('importance', 'N/A')}/10")
        print(f"HEADLINE: {article['headline']}")
        print(f"LINK: {article['link']}")
        print(f"SUMMARY: {article['summary'].get('summary', 'N/A')}")
        print(f"TIME: {article['summary'].get('time', 'N/A')}")
        print(f"CATEGORY: {article['summary'].get('category', 'N/A')}")
        print(f"KEYWORDS: {', '.join(article['summary'].get('keywords', []))}")
        
    # Call the refactored daily news summary generation function
    if processed_articles: # Only run if there were new articles processed in this run
        create_and_save_daily_summary(db, stories_col, topics_col, news_summaries_col)
    else:
        print("Skipping daily summary generation as no new articles were processed in this run.")

    # The process_keyword function might need db and keywords_col passed if it were part of a class
    # For now, as a standalone, it relies on global `db` and `keywords_col` if they were defined globally before main.
    # If not, they need to be passed or made accessible.
    # Assuming it's fine for now or will be refactored separately if needed.

# def process_keyword(keyword: str): # Definition remains, ensure its dependencies are met.
#     print(f'processing keyword: {keyword}')
#     k = keywords_col.find_one({"keyword": keyword})
#     if k is not None:
#         print(f"Keyword {keyword} already exists")
#         return
#     embedding = get_text_embeddings(keyword)
#     keywords_col.insert_one({"keyword": keyword, "embedding": embedding})

