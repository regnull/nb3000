import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
class ChristianScienceMonitor:
    def __init__(self):
        self.url = "https://www.csmonitor.com/layout/set/text/textedition"
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

            print(full_url)
            pattern = r"https://www\.csmonitor\.com/layout/set/text/texteditionlayout/set/text/[A-Za-z]+/[A-Za-z]+/\d{4}/\d{4}/[a-z0-9-]+"
            if re.match(pattern, full_url):
                parsed_url = parse_csm_url(full_url)
                corrected_url = f'https://www.csmonitor.com/text_edition/{parsed_url["section"]}/{parsed_url["subsection"]}/{parsed_url["year"]}/{parsed_url["month_day"]}/{parsed_url["slug"]}'
                articles.append({
                    "headline": headline,
                    "link": corrected_url,
                    "source": "CSM",
                    "updated": parsed_url["date"]
                })

        self.articles = articles
        return articles

def parse_csm_url(url: str) -> dict:
    """
    Parse CSM URL into its component parts.
    
    Args:
        url: URL like /World/Europe/2025/0128/russia-sanctions-fashion-industry-economy
    
    Returns:
        dict with section, subsection, year, month_day, and slug
    """
    pattern = r"https://www\.csmonitor\.com/layout/set/text/texteditionlayout/set/text/([A-Za-z]+)/([A-Za-z]+)/(\d{4})/(\d{4})/([a-z0-9-]+)"
    match = re.match(pattern, url)
    if not match:
        return None
    
    return {
        "section": match.group(1),      # e.g. "World"
        "subsection": match.group(2),   # e.g. "Europe" 
        "year": match.group(3),         # e.g. "2025"
        "month_day": match.group(4),    # e.g. "0128"
        "date": parse_csm_date(match.group(3), match.group(4)),
        "slug": match.group(5)          # e.g. "russia-sanctions-fashion-industry-economy"
    }

def parse_csm_date(year: str, month_day: str) -> datetime:
    """
    Convert CSM URL date components into a datetime object.
    
    Args:
        year: Year as string (e.g. "2025")
        month_day: Month and day as string (e.g. "0128")
    
    Returns:
        datetime object representing the date
    
    Example:
        parse_csm_date("2025", "0128") -> datetime(2025, 1, 28)
    """
    month = int(month_day[:2])
    day = int(month_day[2:])
    return datetime(int(year), month, day)


def main():
    csm = ChristianScienceMonitor()
    articles = csm.fetch_articles()
    print(articles)

if __name__ == "__main__":
    main()  