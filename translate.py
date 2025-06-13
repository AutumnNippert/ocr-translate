#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
translate.py – free German→English multi-lexicon with cache
"""

import json, os, re, shelve, sys, requests, html
from typing import List, Dict
from functools import lru_cache

CACHE = shelve.open(os.path.expanduser("~/.g2e_cache.db"))

WORD_RE = re.compile(r"\w+", re.U)
def clean(t: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(t.strip()))

# ----------------------------------------------------------- PyMultiDictionary
# def fetch_pymulti(word: str) -> List[Dict]:
#     """
#     Use the free PyMultiDictionary wrapper around Linguee/Reverso.
#     Returns up to 5 English glosses for a German word.
#     """
#     try:
#         from PyMultiDictionary import MultiDictionary, DICT_MW          # pip install PyMultiDictionary
#     except ImportError:
#         print("Failed to work")
#         return []

#     try:
#         dic = MultiDictionary()
#         results = dic.meaning('de', word, dictionary=DICT_MW)
#         results = dic.meaning('de', word)
#         print(results)
#     except Exception:
#         return []
def fetch_pymulti(word: str) -> List[Dict]:
    """
    Free wrapper around Linguee/Reverso (pip install PyMultiDictionary).
    Normalises all return shapes into our {pos, definition, examples, source}.
    """
    try:
        from PyMultiDictionary import MultiDictionary, DICT_MW
    except ImportError:
        return []

    try:
        dic = MultiDictionary()
        # MW = Merriam-Webster ; falls back to generic sources internally
        raw = dic.meaning('de', word, dictionary=DICT_MW)
    except Exception:
        return []

    senses: List[Dict] = []

    # Case A – dict: {'Noun': ['street', 'road'], 'Verb': [...] }
    if isinstance(raw, dict):
        for pos, defs in raw.items():
            for gloss in defs[:5]:
                gloss = clean(gloss)
                if gloss:
                    senses.append({
                        "pos": pos.lower(),
                        "definition": gloss,
                        "examples": [],
                        "source": "pymultidict"
                    })

    # Case B – tuple/list format (older API or fallback):
    # ([pos list], long_def, *rest)
    elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
        pos_list, long_def = raw[:2]
        gloss = clean(long_def.split(",")[0])
        for pos in (pos_list if isinstance(pos_list, (list, tuple)) else [pos_list]):
            if gloss:
                senses.append({
                    "pos": str(pos).lower(),
                    "definition": gloss,
                    "examples": [],
                    "source": "pymultidict"
                })
    return senses[:5]
    


# ---------------------------------------------------------------- Wiktionary (pip package)
def fetch_wiktionary(word: str) -> List[Dict]:
    """
    Use the wiktionaryparser package.
    It downloads the *German* page and returns the English translations that
    live under each sense's `translations -> english` list.
    """
    from wiktionaryparser import WiktionaryParser

    parser = WiktionaryParser()

    try:
        page = parser.fetch(word, "german")          # list[dict]
    except Exception:
        return []

    out: List[Dict] = []
    for entry in page:
        for definition in entry.get("definitions", []):
            pos = definition.get("partOfSpeech", "")
            # the bilingual table is keyed by language name
            en_trans = definition.get("translations", {}).get("english", [])
            for tr in en_trans:
                gloss = clean(tr["text"])
                if gloss:
                    out.append({
                        "pos": pos,
                        "definition": gloss,
                        "examples": [],
                        "source": "wiktionary"
                    })
    return out


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

# ---------------------------------------------------------------- orchestrator
def translate(word: str) -> Dict:
    if word in CACHE:
        return CACHE[word]

    senses = ( 
               fetch_pymulti(word) +
               fetch_mymemory(word) +
               fetch_wiktionary(word) +
               fetch_word2word(word) )

    # dedupe
    seen, uniq = set(), []
    for s in senses:
        k = (s["definition"].lower(), s["pos"])
        if k not in seen:
            seen.add(k); uniq.append(s)

    primary = uniq[0]["definition"] if uniq else "—"
    result = dict(word=word, primary=primary,
                  senses=uniq[:12], sources=sorted({s["source"] for s in uniq}))
    CACHE[word] = result
    return result

# ---------------------------------------------------------------- CLI
if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read().strip()
    if not w:
        sys.exit("Give me a German word.")
    print(json.dumps(translate(w), ensure_ascii=False, indent=2))
