#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
translate.py – free German→English multi-lexicon with cache
"""
from time import time
from datetime import datetime
import sys, os
sys.path.append(os.path.dirname(__file__))  # Ensure src/ is in sys.path

from helpers.logging import debug

debug("Starting translation module...")
debug("Importing libraries...")

import json, re, shelve, requests, html
from typing import List, Dict
from functools import lru_cache
import concurrent.futures
import string

# debug("Importing Spacy...")
# # import spacy
# debug("Spacy imported.")
debug("Libraries imported.")

CACHE = shelve.open(os.path.expanduser("~/.g2e_cache.db"))

WORD_RE = re.compile(r"\w+", re.U)
def clean(t: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(t.strip()))

def normalize_word(word: str) -> str:
    # Remove punctuation and lowercase
    return word.strip(string.punctuation + "„“”’'\"").lower()

def lemmatize(word: str) -> str:
    return word
    # Lazy-load the model
    if not hasattr(lemmatize, "_nlp"):
        t_spacy = time()
        lemmatize._nlp = spacy.load("de_core_news_sm")
        debug(f"spaCy model loaded in {time() - t_spacy:.2f}s")
    doc = lemmatize._nlp(word)
    return doc[0].lemma_

# ----------------------------------------------------------- MyMemory (as before)
def fetch_mymemory(word: str) -> List[Dict]:
    out = []
    try:
        js = requests.get("https://api.mymemory.translated.net/get",
                          params=dict(q=word, langpair="de|en"), timeout=4).json()
        prim = js.get("responseData", {}).get("translatedText", "")
        if prim:
            for m in js.get("matches", []):
                if float(m.get("quality", 0)) < 40:       # lower bar
                    continue
                seg = m.get("segment"); trn = m.get("translation")
                if seg and trn:
                    out.append({
                        "pos": "",
                        "definition": clean(trn),
                        "examples": [f"{seg} → {trn}"],
                        "source": "mymemory"
                    })

    except Exception:
        pass
    return out

# ----------------------------------------------------------- word2word offline
@lru_cache(maxsize=4096)
def fetch_word2word(word: str) -> List[Dict]:
    try:
        from word2word import Word2word
        w2w = Word2word("de", "en")
        gloss = w2w(word.lower())[0]
        return [{"pos":"", "definition":gloss, "examples":[],
                 "source":"word2word"}]
    except Exception:
        return []

def fetch_libretranslate(word: str) -> List[Dict]:
    try:
        resp = requests.post(
            "https://libretranslate.com/translate",
            data={"q": word, "source": "de", "target": "en"},
            timeout=4
        )
        trn = resp.json().get("translatedText", "")
        if trn:
            return [{"pos": "", "definition": trn, "examples": [], "source": "libretranslate"}]
    except Exception:
        pass
    return []

# ---------------------------------------------------------------- orchestrator
def translate(word: str) -> Dict:
    t_start = time()
    t_norm = time()
    norm_word = normalize_word(word)
    debug(f"normalize_word: {norm_word} ({time() - t_norm:.3f}s)")

    t_lemma = time()
    lemma_word = lemmatize(norm_word)
    debug(f"lemmatize: {lemma_word} ({time() - t_lemma:.3f}s)")

    # Try cache for original, normalized, or lemmatized
    t_cache = time()
    for key in (word, norm_word, lemma_word):
        if key in CACHE:
            debug(f"cache hit for '{key}' ({time() - t_cache:.3f}s since cache check started)")
            debug(f"total translate() time: {time() - t_start:.3f}s")
            return CACHE[key]
    debug(f"cache miss ({time() - t_cache:.3f}s)")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        def timed_fetch(name, func, arg):
            start = time()
            result = func(arg)
            debug(f"{name} returned: {result}")
            debug(f"Fetched {name} in {time() - start:.2f} seconds")
            return result

        t_fetch = time()
        futures = [
            executor.submit(timed_fetch, "MyMemory", fetch_mymemory, lemma_word),
            executor.submit(timed_fetch, "Word2Word", fetch_word2word, lemma_word),
            executor.submit(timed_fetch, "LibreTranslate", fetch_libretranslate, lemma_word),
        ]
        senses = []
        for fut in futures:
            try:
                senses += fut.result(timeout=5)
            except Exception:
                pass
        debug(f"All fetches done in {time() - t_fetch:.2f} seconds")


    primary = senses[0]["definition"] if senses else "—"
    result = dict(word=word, primary=primary,
                  senses=senses[:12], sources=sorted({s["source"] for s in senses}))
    # Cache under all forms for fast future lookup
    t_cache_write = time()
    for key in (word, norm_word, lemma_word):
        CACHE[key] = result
    debug(f"Cache write done in {time() - t_cache_write:.3f}s")
    debug(f"total translate() time: {time() - t_start:.3f}s")
    return result

# ---------------------------------------------------------------- CLI
if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read().strip()
    if not w:
        sys.exit("Give me a German word.")
    print(json.dumps(translate(w), ensure_ascii=False, indent=2))