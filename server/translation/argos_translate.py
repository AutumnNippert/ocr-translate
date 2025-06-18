
import time
import argostranslate.package, argostranslate.translate

print("Initializing ArgosTranslate...")
start = time.time()
# Download and install languages
argostranslate.package.update_package_index()
packages = argostranslate.package.get_available_packages()
[package] = [p for p in packages if p.from_code == "de" and p.to_code == "en"]
argostranslate.package.install_from_path(package.download())
end = time.time()
print(f"ArgosTranslate packages updated in {end - start:.2f} seconds")

import dbm
import json
import os
from HanTa import HanoverTagger as ht

tagger_de = ht.HanoverTagger('morphmodel_ger.pgz')

CACHE_DB_FILE = '.wordcache.db'

# check if the cache file exists, if not create it
if not os.path.exists(CACHE_DB_FILE):
    with dbm.open(CACHE_DB_FILE, 'c') as db:
        # Initialize the database if it doesn't exist
        db['initialized'] = json.dumps({'version': 1}).encode()
        print(f"Cache database '{CACHE_DB_FILE}' created.")

def db_open(mode='c'):
    return dbm.open(CACHE_DB_FILE, mode)

def cache_get(word: str):
    with db_open('r') as db:
        if word.encode() in db:
            return json.loads(db[word.encode()].decode())
    return None

def cache_set(word: str, value: dict):
    with db_open('c') as db:
        db[word.encode()] = json.dumps(value).encode()

def cache_size():
    with db_open('r') as db:
        return len(db)

start = time.time()
entries = cache_size()
end = time.time()
print(f"Cache opened in {end - start:.2f} seconds, {entries} entries found.")

def translate(word: str) -> dict:
    word = word.lower()
    pos = tagger_de.analyze(word)
    print(f"POS tags for '{word}': {pos}")

    cached = cache_get(word)
    if cached:
        print(f"Cache hit for word '{word}'")
        return cached
    else:
        print(f"Cache miss for word '{word}'")

    if word.isdigit():
        print(f"Word '{word}' is a number, returning as is.")
        result_dict = {
            'word': word,
            'definitions': {'NUM': word},
            'html': f'<span class="translation">{word}</span>'
        }
        cache_set(word, result_dict)
        return result_dict

    result = argostranslate.translate.translate(word, "de", "en")
    print(result)
    result_dict = {
        'word': word,
        'definitions': {
            pos[1]: pos[0] + " -> " + result
        },
        'html': f'<span class="translation">{result}</span>'
    }

    cache_set(word, result_dict)
    return result_dict

if __name__ == "__main__":
    starttime = time.time()
    result = translate("Super Kuhl!")
    endtime = time.time()
    print(f"Translation took {endtime - starttime:.2f} seconds")
    print(f"Result: {result}")
    print(f"Values: {result['definitions'].values()}")
    for pos, defs in result['definitions'].items():
        print(f"POS: {pos}, Definitions: {defs}")
