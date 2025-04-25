import requests
from bs4 import BeautifulSoup
import re

class AssociatedPress:
    def __init__(self):
        self.url = "https://apnews.com/"
        self.articles = []

    def fetch_articles(self):
        """
        Fetch and filter Associated Press article links.
        
        Returns:
            list: List of dictionaries containing article headline and URL
        """
        url = "https://apnews.com/"
        
        try:
            # Add headers to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            articles = []

            # Find all article links
            for link in soup.find_all('a', href=True):
                href = link['href']
                headline = link.get_text(strip=True)

                # Reuters links are often relative
                full_url = url + href if href.startswith('/') else href
                
                # Match Reuters article pattern
                pattern = r'https://apnews\.com/article/[a-z0-9-]+'
                if re.match(pattern, full_url) and headline:
                    articles.append({
                        "headline": headline,
                        "link": full_url,
                        "source": "Associated Press"
                    })

            # Remove duplicate articles by using a set to track unique URLs
            unique_urls = set()
            unique_articles = []
            
            for article in articles:
                if article['link'] not in unique_urls:
                    unique_urls.add(article['link'])
                    unique_articles.append(article)
            
            articles = unique_articles
            return articles

        except requests.exceptions.RequestException as e:
            print(f"Error fetching Associated Press content: {e}")
            return []

def main():
    ap = AssociatedPress()
    articles = ap.fetch_articles()
    for article in articles:
        print(f"Headline: {article['headline']}")
        print(f"URL: {article['link']}")
        print("---")

if __name__ == "__main__":
    main()
