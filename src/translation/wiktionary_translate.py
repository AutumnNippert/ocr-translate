import requests
from bs4 import BeautifulSoup, NavigableString
import html

from HanTa import HanoverTagger as ht
tagger_de = ht.HanoverTagger('morphmodel_ger.pgz')

PARTS_OF_SPEECH = [
    "noun", "verb", "adjective", "adverb", "pronoun", "determiner",
    "preposition", "postposition", "conjunction", "interjection", 
    "particle", "numeral", "symbol", "suffix", "prefix", "infix",
    "circumfix", "proper noun", "phrase", "idiom", "proverb", 
    "letter", "punctuation", "contraction", "expression"
]

CACHE_LOCATION = "./wiki_cache"

def get_all_text(soup: BeautifulSoup) -> str:
    try:
        text_parts = []
        for element in soup.descendants:
            if isinstance(element, NavigableString):
                stripped = element.strip()
                if stripped:
                    text_parts.append(stripped)
        return html.unescape(" ".join(text_parts))
    except Exception as e:
        print(f"Error while cleaning {soup.name}: {e}")
        return None

def get_word(word: str, language: str = "German") -> dict:
    cache_filename = f"{CACHE_LOCATION}/{word}.html"
    import os
    if not os.path.exists("./wiki_cache"):
        os.makedirs("./wiki_cache")
    # try:
    #     with open(cache_filename, "r", encoding="utf-8") as cache_file:
    #         return cache_file.read()
    # except FileNotFoundError:
    #     pass

    url = f"https://en.wiktionary.org/api/rest_v1/page/html/{word}"
    response = requests.get(url)
    if response.status_code != 200:
        return ""
    
    soup = BeautifulSoup(response.text, "html.parser")
    # get list of all sections
    sections = soup.find_all("section")
    german_section = None
    for section in sections:
        if section.find("h2") and language in section.find("h2").text:
            german_section = section
            break
    
    if not german_section:
        print(f"No section found for language: {language}")
        return None
    # get all subsections of the german section
    data = {}
    subsections = german_section.find_all("section")
    for subsection in subsections:
        header = subsection.find("h3")
        if header:
            contents = header.contents[0]
            # Only clean headers that are not part of the parts of speech
            if str(contents).lower() not in PARTS_OF_SPEECH:
                print(f'Cleaning of header: {header.text.strip()}')
                data[header.text.strip()] = get_all_text(subsection)
                continue
            siblings = []
            for sibling in header.next_siblings:
                if sibling.name == "p":
                    continue # fancy other forms
                siblings.append(sibling)
                if sibling.name == "section":
                    break

            header_text = header.text.strip()
            data[header_text] = siblings

    definitions = {}
    for part_of_speech in PARTS_OF_SPEECH:
        for key in data.keys():
            if part_of_speech == key.lower():
                siblings = data[key]
                word_defs = []

                for sibling in siblings:
                    if sibling.name == "ol":
                        for subsibling in sibling.find_all("li", recursive=False):
                            children = subsibling.contents
                            definition_parts = []

                            for child in children:
                                if getattr(child, "name", None) == "dl":
                                    break  # Stop before <dl>
                                if isinstance(child, str):
                                    definition_parts.append(child.strip())
                                else:
                                    definition_parts.append(child.get_text(strip=True))

                            definition = " ".join(definition_parts).strip()
                            # <dl> is other information regarding the definition
                            examples = subsibling.find("dl")
                            word_def = {
                                "definition": definition if definition else "",
                                "extra": [ex.text for ex in examples.find_all("dd")] if examples else []
                            }
                            word_defs.append(word_def)
                definitions[key] = word_defs
                print(f'Cleaning of part of speech: {key}')
                data[key] = get_all_text(sibling) # clean up the PoS part of the text
    if not definitions:
        print(f"No definitions found for word: {word}")
        return None

    data['definitions'] = definitions
    data['html'] = str(german_section.decode_contents())

    # add to cache
    with open(cache_filename, "w", encoding="utf-8") as cache_file:
        cache_file.write(str(german_section))
    return data

def translate(word: str) -> str:
    lemma = tagger_de.analyze(word)[0]
    return get_word(lemma, 'German')

if __name__ == "__main__":
    word = "Dummkopf"  # Example word to translate
    translation = translate(word)
    # print(translation)
    if translation:
        print(f"Translation for '{word}':")
        print(translation['definitions'])

        # remve the html key from the translation
        translation.pop('html', None)

        print(translation)

        # print the translation in a readable format using json
        import json
        print(json.dumps(translation, indent=4, ensure_ascii=False))

    else:
        print(f"No translation found for '{word}'.")