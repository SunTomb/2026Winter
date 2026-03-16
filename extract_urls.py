import os
import fitz
import re
import glob

files = glob.glob('**/*.pdf', recursive=True)
arxiv_pattern = re.compile(r'arxiv(?:\.org/abs/|:)(\d{4}\.\d{4,5})', re.IGNORECASE)
url_pattern = re.compile(r'https?://[^\s]+')

for f in files:
    try:
        doc = fitz.open(f)
        text = ""
        for page in doc[:3]: # check first 3 pages
            text += page.get_text()
        
        arxiv_match = arxiv_pattern.search(text)
        if arxiv_match:
            print(f"{f}: https://arxiv.org/abs/{arxiv_match.group(1)}")
        else:
            # Let's search for other URLs like github maybe
            urls = url_pattern.findall(text)
            arxiv_urls = [u for u in urls if 'arxiv.org' in u]
            if arxiv_urls:
                print(f"{f}: {arxiv_urls[0]}")
            else:
                print(f"{f}: NO URL FOUND. found: {urls[:3]}")
    except Exception as e:
        print(f"Error on {f}: {e}")
