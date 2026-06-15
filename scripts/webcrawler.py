import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
from collections import deque

START_URL = "https://www.csusb.edu/graduate-studies"

# Keep only pages from same website
ALLOWED_DOMAIN = urlparse(START_URL).netloc

# Keep only Graduate Studies pages.
# Change this to "/" if you want the full csusb.edu website, but that may be too much.
ALLOWED_PATH_PREFIX = "/graduate-studies"

MAX_PAGES = 150

IGNORE_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg",
    ".zip", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 Academic Chatbot URL Collector"
}


def clean_url(url):
    url, _ = urldefrag(url)
    return url.rstrip("/")


def is_valid_url(url):
    parsed = urlparse(url)

    if parsed.scheme not in ["http", "https"]:
        return False

    if parsed.netloc != ALLOWED_DOMAIN:
        return False

    if not parsed.path.startswith(ALLOWED_PATH_PREFIX):
        return False

    if parsed.path.lower().endswith(IGNORE_EXTENSIONS):
        return False

    return True


def collect_urls():
    visited = set()
    queue = deque([START_URL])

    while queue and len(visited) < MAX_PAGES:
        current_url = clean_url(queue.popleft())

        if current_url in visited:
            continue

        print(f"Checking: {current_url}")
        visited.add(current_url)

        try:
            response = requests.get(current_url, headers=HEADERS, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"Skipped: {current_url} | {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        for link in soup.find_all("a", href=True):
            absolute_url = clean_url(urljoin(current_url, link["href"]))

            if is_valid_url(absolute_url) and absolute_url not in visited:
                queue.append(absolute_url)

    with open("urls.txt", "w", encoding="utf-8") as file:
        file.write("# One URL per line. Lines starting with # are ignored.\n")
        for url in sorted(visited):
            file.write(url + "\n")

    print(f"\nDone. Saved {len(visited)} URLs to urls.txt")


if __name__ == "__main__":
    collect_urls()