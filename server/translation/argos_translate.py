
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
import atexit
import signal
import sys

tagger_de = ht.HanoverTagger('morphmodel_ger.pgz')

CACHE_DB_FILE = '.wordcache.db'

class PersistentCache:
    def __init__(self, path: str):
        # 'c' = create if needed, read-write otherwise
        self._db = dbm.open(path, 'c')

    def get(self, key: str):
        key_b = key.encode()
        if key_b in self._db:
            return json.loads(self._db[key_b].decode())

    def set(self, key: str, value: dict):
        self._db[key.encode()] = json.dumps(value).encode()

    def size(self):
        return len(self._db)

    # -------- graceful shutdown helpers --------
    def flush(self):
        """Write in-memory buffers to disk without closing."""
        if hasattr(self._db, "sync"):     # gdbm & ndbm expose .sync()
            self._db.sync()

    def close(self):
        """Flush and close safely (idempotent)."""
        try:
            self.flush()
        finally:
            self._db.close()

# ------------------------------------------------

def _graceful_exit(signum=None, frame=None):
    # -- give yourself a hint why you're exiting
    reason = f"signal {signum}" if signum else "normal exit"
    print(f"\n[shutdown] saving cache on {reason} …", file=sys.stderr)
    cache.close()
    # If we arrived here from a signal handler, exit explicitly
    if signum is not None:
        sys.exit(0)


# check if the cache file exists, if not create it
if not os.path.exists(CACHE_DB_FILE):
    with dbm.open(CACHE_DB_FILE, 'c') as db:
        # Initialize the database if it doesn't exist
        db['initialized'] = json.dumps({'version': 1}).encode()
        print(f"Cache database '{CACHE_DB_FILE}' created.")

start = time.time()
cache = PersistentCache(CACHE_DB_FILE)
entries = cache.size()
end = time.time()
print(f"Cache opened in {end - start:.2f} seconds, {entries} entries found.")

# normal interpreter shutdown
atexit.register(_graceful_exit)

# (Ctrl-C, kill)
signal.signal(signal.SIGINT,  _graceful_exit)   # Ctrl-C
signal.signal(signal.SIGTERM, _graceful_exit)   # `kill <pid>`

def translate(word: str) -> dict:
    word = word.lower()

    cached = cache.get(word)
    if cached:
        return cached

    # Lemmatize only after (kinda mid but eh thats fine)
    pos = tagger_de.analyze(word)
    print(f"POS tags for '{word}': {pos}")

    if word.isdigit():
        result_dict = {
            'word': word,
            'definitions': {'NUM': word},
            'html': f'<span class="translation">{word}</span>'
        }
        cache.set(word, result_dict)
        return result_dict

    result = argostranslate.translate.translate(word, "de", "en")
    result_dict = {
        'word': word,
        'definitions': {
            pos[1]: pos[0] + " -> " + result
        },
        'html': f'<span class="translation">{result}</span>'
    }

    cache.set(word, result_dict)
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
