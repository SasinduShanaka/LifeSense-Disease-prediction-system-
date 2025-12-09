"""
Web Scraping Utilities
Tools for extracting data from websites
"""
import requests
from urllib.parse import urljoin, urlparse
import json
import time
from datetime import datetime


class WebScraper:
    """Generic web scraper with rate limiting"""
    
    def __init__(self, base_url, delay=1):
        """
        Initialize scraper
        
        Args:
            base_url: Base URL for the website
            delay: Delay between requests in seconds
        """
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_page(self, url, params=None):
        """
        Fetch a web page
        
        Args:
            url: URL to fetch
            params: Query parameters
            
        Returns:
            Response object or None if error
        """
        try:
            time.sleep(self.delay)
            full_url = urljoin(self.base_url, url)
            response = self.session.get(full_url, params=params, timeout=10)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def extract_links(self, html_content):
        """Extract all links from HTML content"""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        
        for anchor in soup.find_all('a', href=True):
            link = anchor['href']
            full_link = urljoin(self.base_url, link)
            links.append(full_link)
        
        return links
    
    def extract_text(self, html_content, selector=None):
        """Extract text content from HTML"""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        if selector:
            elements = soup.select(selector)
            return [elem.get_text(strip=True) for elem in elements]
        
        return soup.get_text(strip=True)
    
    def extract_table(self, html_content, table_index=0):
        """Extract data from HTML table"""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')
        
        if table_index >= len(tables):
            return None
        
        table = tables[table_index]
        data = []
        
        # Extract headers
        headers = []
        header_row = table.find('thead')
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
        
        # Extract rows
        for row in table.find_all('tr'):
            cells = row.find_all(['td', 'th'])
            if cells:
                row_data = [cell.get_text(strip=True) for cell in cells]
                data.append(row_data)
        
        return {'headers': headers, 'data': data}
    
    def extract_images(self, html_content):
        """Extract all image URLs from HTML"""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html_content, 'html.parser')
        images = []
        
        for img in soup.find_all('img', src=True):
            img_url = urljoin(self.base_url, img['src'])
            images.append({
                'url': img_url,
                'alt': img.get('alt', ''),
                'title': img.get('title', '')
            })
        
        return images
    
    def download_file(self, url, save_path):
        """Download file from URL"""
        try:
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False
    
    def save_to_json(self, data, filename):
        """Save scraped data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def crawl(self, start_url, max_pages=10, same_domain=True):
        """
        Crawl website starting from URL
        
        Args:
            start_url: Starting URL
            max_pages: Maximum pages to crawl
            same_domain: Only crawl pages on same domain
        """
        visited = set()
        to_visit = [start_url]
        results = []
        
        domain = urlparse(self.base_url).netloc
        
        while to_visit and len(visited) < max_pages:
            url = to_visit.pop(0)
            
            if url in visited:
                continue
            
            # Check domain restriction
            if same_domain and urlparse(url).netloc != domain:
                continue
            
            print(f"Crawling: {url}")
            response = self.get_page(url)
            
            if response and response.status_code == 200:
                visited.add(url)
                
                # Store page data
                results.append({
                    'url': url,
                    'status': response.status_code,
                    'timestamp': datetime.now().isoformat(),
                    'content_type': response.headers.get('content-type', '')
                })
                
                # Extract links
                if 'text/html' in response.headers.get('content-type', ''):
                    links = self.extract_links(response.text)
                    to_visit.extend([l for l in links if l not in visited])
        
        return results


class APIClient:
    """Client for interacting with REST APIs"""
    
    def __init__(self, base_url, api_key=None):
        """Initialize API client"""
        self.base_url = base_url
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def get(self, endpoint, params=None):
        """GET request"""
        url = urljoin(self.base_url, endpoint)
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def post(self, endpoint, data=None, json_data=None):
        """POST request"""
        url = urljoin(self.base_url, endpoint)
        response = self.session.post(url, data=data, json=json_data)
        response.raise_for_status()
        return response.json()
    
    def put(self, endpoint, data=None, json_data=None):
        """PUT request"""
        url = urljoin(self.base_url, endpoint)
        response = self.session.put(url, data=data, json=json_data)
        response.raise_for_status()
        return response.json()
    
    def delete(self, endpoint):
        """DELETE request"""
        url = urljoin(self.base_url, endpoint)
        response = self.session.delete(url)
        response.raise_for_status()
        return response.json()


def scrape_pagination(scraper, url_template, max_pages=10):
    """
    Scrape paginated content
    
    Args:
        scraper: WebScraper instance
        url_template: URL with {page} placeholder
        max_pages: Maximum pages to scrape
    """
    all_data = []
    
    for page in range(1, max_pages + 1):
        url = url_template.format(page=page)
        response = scraper.get_page(url)
        
        if response and response.status_code == 200:
            # Extract data from page
            data = scraper.extract_text(response.text)
            all_data.append(data)
        else:
            break
    
    return all_data
