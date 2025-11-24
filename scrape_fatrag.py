import os
import subprocess
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque

# Hardcoded lijst van URLs: gegeven + extra relevante van pieterman.com
URLS = [
    # Gegeven URLs
    "https://www.pieterman.com/nvlo-jaarcongres-10-tips-leiderschap-lef/",
    "https://www.pieterman.com/eigenaarschap-verwarren-met-verantwoordelijkheid-voor-de-totale-uitkomst/",
    "https://www.pieterman.com/bakje/",
    "https://www.pieterman.com/5-thuiswerk-tips/",
    "https://www.pieterman.com/ambitieuze-millenials/",
    "https://www.pieterman.com/trainersfouten-tips/",
    "https://www.pieterman.com/veilig-positief-leerklimaat-ok/",
    
    # Extra relevante (gevonden via site-search: trainingen, tips, podcasts, etc.)
    "https://www.pieterman.com/trainingen/succesvol-in-leiderschap/",
    "https://www.pieterman.com/inspiratie-verhalen-en-tips/",
    "https://www.pieterman.com/generatieverschillen-werkvloer-5-tips/",
    "https://www.pieterman.com/podcast-leidinggevende-aanspreken-rolverandering/",
    "https://www.pieterman.com/podcast-pieterman/",
    "https://www.pieterman.com/category/training/",
    "https://www.pieterman.com/trainingen/train-de-trainer/",
    "https://www.pieterman.com/podcast-aflevering-7-normaliseren-negativiteit/",
    # Voeg hier meer toe als je ze vindt, of gebruik de crawler
]

# Output map
OUTPUT_DIR = "./fatrag_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Headers om Cloudflare te omzeilen: Bootst een browser na
CURL_HEADERS = [
    '-H', 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    '-H', 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    '-H', 'Accept-Language: en-US,en;q=0.5',
    '-H', 'Referer: https://www.google.com/',
    '--compressed'  # Voor gzip compressie
]

# Functie om schone tekst te extracten uit HTML
def extract_content(html):
    soup = BeautifulSoup(html, 'html.parser')
    
    # Verwijder onnodige elementen (scripts, styles, nav, footer)
    for element in soup(['script', 'style', 'nav', 'footer', 'header']):
        element.decompose()
    
    # Probeer main content te vinden (typisch <main> or <article>)
    content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.body
    if content:
        text = content.get_text(separator="\n", strip=True)
        # Filter op relevante keywords om ruis te verminderen
        if any(keyword in text.lower() for keyword in ["tips", "leiderschap", "gedrag", "training", "inspiratie", "verhalen", "ambitie", "next step"]):
            return text
    return "Geen relevante content gevonden."

# Functie om een enkele URL te scrapen met curl en op te slaan
def scrape_url(url):
    try:
        # Bouw curl commando
        curl_cmd = ['curl', '-s', '-L'] + CURL_HEADERS + [url]
        response = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=10)
        response.check_returncode()  # Raise als niet succesvol
        html = response.stdout
        content = extract_content(html)
        
        # Bestandsnaam baseren op URL-slug
        slug = urlparse(url).path.strip('/').replace('/', '_') or 'homepage'
        filename = f"{slug}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Opgeslagen: {filepath}")
        
        # Delay om rate limiting te vermijden
        time.sleep(2)  # 2 seconden pauze per request
    except subprocess.TimeoutExpired:
        print(f"Timeout bij {url}")
    except subprocess.CalledProcessError as e:
        print(f"Fout bij {url}: {e.returncode} - {e.stderr}")
    except Exception as e:
        print(f"Algemene fout bij {url}: {e}")

# Basis crawler: Start vanaf een URL, volg same-domain links (beperkt diepte)
def crawl(start_url, max_depth=2, max_pages=50):
    visited = set()
    queue = deque([(start_url, 0)])  # (url, depth)
    domain = urlparse(start_url).netloc
    
    while queue and len(visited) < max_pages:
        url, depth = queue.popleft()
        if url in visited or depth > max_depth:
            continue
        visited.add(url)
        scrape_url(url)
        
        try:
            # Haal HTML met curl
            curl_cmd = ['curl', '-s', '-L'] + CURL_HEADERS + [url]
            response = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=10)
            soup = BeautifulSoup(response.stdout, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)
                if urlparse(full_url).netloc == domain and full_url not in visited:
                    queue.append((full_url, depth + 1))
            time.sleep(2)  # Delay in crawler
        except Exception as e:
            print(f"Crawl fout bij {url}: {e}")

# Main: Scrape alle hardcoded URLs, dan optioneel crawl
if __name__ == "__main__":
    print("Start scraping van hardcoded URLs met curl...")
    for url in URLS:
        scrape_url(url)
    
    # Optioneel: Crawl vanaf een startpunt voor meer data (bijv. categorie)
    # Uncomment en pas aan als nodig
    # print("\nStart crawling vanaf categorie...")
    # crawl("https://www.pieterman.com/category/training/")
    
    print("\nKlaar! Data opgeslagen in", OUTPUT_DIR)
