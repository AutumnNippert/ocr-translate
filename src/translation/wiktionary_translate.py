import requests
from bs4 import BeautifulSoup
from HanTa import HanoverTagger as ht
tagger_de = ht.HanoverTagger('morphmodel_ger.pgz')

CACHE_LOCATION = "./wiki_cache"

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
            siblings = []
            for sibling in header.next_siblings:
                if sibling.name == "section":
                    break
                siblings.append(sibling)
            header_text = header.text.strip()
            data[header_text] = "\n".join(str(sibling.text) for sibling in siblings)
    
    data['html'] = str(german_section.decode_contents())

    # print the data keys
    for key in data.keys():
        print(f"Key: {key}")

    # add to cache
    with open(cache_filename, "w", encoding="utf-8") as cache_file:
        cache_file.write(str(german_section))
    return data

def translate(word: str) -> str:
    lemma = tagger_de.analyze(word)[0]
    return get_word(lemma, 'German')

if __name__ == "__main__":
    word = "Haus"
    translation = translate(word)
    # print(translation)
    if translation:
        print(f"Translation for '{word}':")
        print(translation.get('Noun', 'No HTML content found.'))
    else:
        print(f"No translation found for '{word}'.")