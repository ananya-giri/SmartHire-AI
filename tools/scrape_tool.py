import requests
from bs4 import BeautifulSoup
from crewai.tools import tool

@tool("scrape_website")
def scrape_website(url: str) -> str:
    """Useful to scrape and extract text from a given webpage URL. 
    Use this to read specific candidate profiles, GitHub portfolios or job boards."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.extract()
            
        text = soup.get_text(separator=' ', strip=True)
        if not text:
            return f"[SCRAPE_ERROR: No text content found at URL: {url}]"
            
        return text[:4000]
    except Exception as e:
        return f"[SCRAPE_ERROR: Perception layer failed to fetch {url}: {str(e)}]"

def get_scrape_tool():
    return scrape_website
