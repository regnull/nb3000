import requests
from bs4 import BeautifulSoup
import re
from pymongo.mongo_client import MongoClient
from datetime import datetime
import os
import os
from llm import summarize_article, get_text_embeddings
from dotenv import load_dotenv
from csm import ChristianScienceMonitor

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
            pattern = r"https://lite\.cnn\.com/\d{4}/\d{2}/\d{2}/[a-z-]+/[a-z0-9-]+/index\.html"
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
                timestamp_text = timestamp_text.replace(" EST", "").strip()
                timestamp_format = "%I:%M %p, %a %B %d, %Y"
                parsed_timestamp = datetime.strptime(timestamp_text, timestamp_format)

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

    csm = ChristianScienceMonitor()

    run_start_time = datetime.now()
    articles = fetch_cnn_lite_content()
    articles.extend(csm.fetch_articles())
    mongo_uri = os.getenv("MONGO_URI")
    client = MongoClient(mongo_uri)
    db = client.get_database('nb3000')
    stories_col = db.get_collection('stories')

    print("Fetching and analyzing stories...")
    processed_articles = []
    for article in articles:
        print(f"Processing article {article['headline']}...")

        # If an article with the same headline exists, we skip.
        # At some point, we might want to update the article if the timestamp is different.
        if stories_col.find_one({ "headline": article["headline"] }):
            print("article exists, skipping")
            continue

        text, timestamp = fetch_url_text(article['link'], parse_timestamp=(article["source"] == "CNN"))
        if timestamp:
            article['updated'] = timestamp

        summary = summarize_article(text)
        if 'Error' in summary:
            print(f"Error processing article {article['link']}: {summary}")
            continue

        article['summary'] = summary
        article['run_start_time'] = run_start_time

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

    stories_col.insert_many(processed_articles)

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

