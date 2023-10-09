import urllib.request
import fitz
import re
import openai
import os

from src.semantic_search import SemanticSearch

def download_pdf(url, output_path):
    urllib.request.urlretrieve(url, output_path)

def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text

def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page-1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list

def text_to_chunks(texts, word_length=150, start_page=1):
    text_toks = [t.split(' ') for t in texts]
    page_nums = []
    chunks = []
    
    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i:i+word_length]
            if (i+word_length) > len(words) and (len(chunk) < word_length) and (
                len(text_toks) != (idx+1)):
                text_toks[idx+1] = chunk + text_toks[idx+1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[Page no. {idx+start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


def load_recommender(path, start_page=1):
    recommender = SemanticSearch()
    texts = pdf_to_text(path, start_page=start_page)
    chunks = text_to_chunks(texts, start_page=start_page)
    recommender.fit(chunks)
    return recommender


def generate_text(openAI_key,prompt, engine="text-davinci-003"):
    openai.api_key = openAI_key
    completions = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.7,
    )
    message = completions.choices[0].text
    return message





def sanitize_filename(filename):
    # Strip file extension
    name, ext = filename.rsplit('.', 1)
    
    # Remove characters that aren't in [a-zA-Z0-9_-]
    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '', name)
    
    # Ensure the sanitized name is up to 64 characters long
    sanitized_name = sanitized_name[:64]
    
    # If for any reason the sanitized name is empty, give it a default
    if not sanitized_name:
        sanitized_name = "default"
    
    # Return sanitized name with original file extension
    return f"{sanitized_name}.{ext}".replace("-", "_").replace("_", "")[:12]


if __name__ == "__main__":
    # Test
    original_name = "A complex & weird-name for_a.PDF!!!.pdf"
    print(sanitize_filename(original_name))
