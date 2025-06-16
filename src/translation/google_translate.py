from google.cloud import translate_v2 as gtranslate_v2
client = gtranslate_v2.Client()

from HanTa import HanoverTagger as ht
tagger_de = ht.HanoverTagger('morphmodel_ger.pgz')

import pickle

cache = {}
# load .wordcache file if it exists
import os
import time

def load_cache():
    global cache
    if os.path.exists('.wordcache'):
        with open('.wordcache', 'rb') as f:
            cache = pickle.load(f)

def save_cache():
    global cache
    with open('.wordcache', 'wb') as f:
        pickle.dump(cache, f)

start = time.time()
load_cache()
end = time.time()
print(f"Cache loaded in {end - start:.2f} seconds, {len(cache)} entries found.")

def translate(word: str) -> dict:
    word = word.lower()
    pos = tagger_de.analyze(word)
    print(f"POS tags for '{word}': {pos}")
    if word in cache:
        print(f"Cache hit for word '{word}'")
        return cache[word]
    
    # if is numbers
    if word.isdigit():
        print(f"Word '{word}' is a number, returning as is.")
        result_dict = {
            'word': word,
            'definitions': {'NUM': word},
            'html': f'<span class="translation">{word}</span>'
        }
        cache[word] = result_dict
        save_cache()
        return result_dict

    result = client.translate(word, source_language="de", target_language="en")
    print(result)
    result_dict = {
        'word': word,
        'definitions': {
            pos[1]: pos[0] + " -> " + result["translatedText"]
        },
        'html': f'<span class="translation">{result["translatedText"]}</span>'
    }

    cache[word] = result_dict
    save_cache()
    return result_dict

if __name__ == "__main__":
    starttime = time.time()
    result = translate("Das ist ein Test")
    endtime = time.time()
    print(f"Translation took {endtime - starttime:.2f} seconds")
    print(f"Result: {result}")
    print(f"values: {result['definitions'].values()}")
    for pos, defs in result['definitions'].items():
        print(f"POS: {pos}, Definitions: {defs}")
