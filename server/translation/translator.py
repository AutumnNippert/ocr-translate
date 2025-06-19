import dbm
import json
import os
import atexit
import signal
import sys

from translation.wiktionary_translate import translate as wiktionary_translate
from translation.argos_translate import translate as argos_translate
from translation.google_translate import translate as google_translate

from HanTa import HanoverTagger as ht
tagger_de = ht.HanoverTagger('morphmodel_ger.pgz')

CACHE_DB_FILE = '.wordcache.db'

class PersistentCache:
    def __init__(self, path: str):
        self._db = dbm.open(path, 'c')

    def get(self, key: str):
        key_b = key.encode()
        if key_b in self._db:
            return json.loads(self._db[key_b].decode())

    def set(self, key: str, value: dict):
        self._db[key.encode()] = json.dumps(value).encode()

    def size(self):
        return len(self._db)

    def flush(self):
        if hasattr(self._db, "sync"):
            self._db.sync()

    def close(self):
        try:
            self.flush()
        finally:
            self._db.close()

def _graceful_exit(signum=None, frame=None):
    reason = f"signal {signum}" if signum else "normal exit"
    print(f"\n[shutdown] saving cache on {reason} …", file=sys.stderr)
    cache.close()
    if signum is not None:
        sys.exit(0)

# Ensure cache file exists
if not os.path.exists(CACHE_DB_FILE):
    with dbm.open(CACHE_DB_FILE, 'c') as db:
        db['initialized'] = json.dumps({'version': 1}).encode()
        print(f"Cache database '{CACHE_DB_FILE}' created.")

cache = PersistentCache(CACHE_DB_FILE)
atexit.register(_graceful_exit)
signal.signal(signal.SIGINT,  _graceful_exit)
signal.signal(signal.SIGTERM, _graceful_exit)

def translate(word: str) -> dict:
    """
    Try to translate using Wiktionary, then Argos, then Google Translate.
    Always return in the Argos-style format:
    {
        'word': ...,
        'definitions': { POS: "lemma -> translation" },
        'html': ...
    }
    """
    word_key = word.lower()
    cached = cache.get(word_key)
    if cached:
        return cached

    lemma, pos = tagger_de.analyze(word)
    print(f"POS tags for '{word}': {pos}")

    # Try Wiktionary first
    wiktionary_result = wiktionary_translate(lemma, pos)
    if wiktionary_result and isinstance(wiktionary_result, dict) and 'definitions' in wiktionary_result:
        definitions = {}
        for pos, defs in wiktionary_result['definitions'].items():
            if defs and isinstance(defs, list):
                joined_defs = "; ".join(d['definition'] for d in defs if 'definition' in d)
                if joined_defs:
                    definitions[pos] = joined_defs
        html = wiktionary_result.get('html', '')
        result = {
            'word': word,
            'definitions': definitions,
            'html': html
        }
        cache.set(word_key, result)
        return result

    if word.isdigit():
        return {
            'word': word,
            'definitions': {'NUM': word},
            'html': f'<span class="translation">{word}</span>'
        }
    
    # Try Argos next
    argos_result = argos_translate(word, pos)
    if argos_result and isinstance(argos_result, dict) and 'definitions' in argos_result:
        import pprint
        print(f"Cached result for '{word}': {pprint.pformat(argos_result)}")
        cache.set(word_key, argos_result)
        return argos_result

    # Fallback to Google Translate
    google_result = google_translate(word, pos)
    if google_result and isinstance(google_result, dict) and 'definitions' in google_result:
        cache.set(word_key, google_result)
        return google_result

    # If all fail
    result = {
        'word': word,
        'definitions': {},
        'html': '<span class="translation">No translation found</span>'
    }
    cache.set(word_key, result)
    return result