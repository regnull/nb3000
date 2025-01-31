import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime

class NPR:
    def __init__(self):
        self.url = "https://text.npr.org/"
        self.articles = []

    def fetch_articles(self):
        response = requests.get(self.url)
        response.raise_for_status()

        # Parse the page with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = []
        for link in soup.find_all('a', href=True):
            headline = link.get_text(strip=True)
            href = link['href']

            full_url = self.url + href.lstrip('/') if href.startswith("/") else href

            # Match NPR text article URLs (e.g. https://text.npr.org/nx-s1-5281946)
            pattern = r"https://text\.npr\.org/nx-s1-\d+"
            if re.match(pattern, full_url):
                articles.append({
                    "headline": headline,
                    "link": full_url,
                    "source": "NPR"
                })

        self.articles = articles
        return articles

def main():
    npr = NPR()
    articles = npr.fetch_articles()
    print(articles)

if __name__ == "__main__":
    main()  