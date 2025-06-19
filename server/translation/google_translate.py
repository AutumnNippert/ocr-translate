import time
from google.cloud import translate_v2 as gtranslate_v2

print("Initializing Google Translate client...")
client = gtranslate_v2.Client()
print("Google Translate client initialized.")

def translate(lemma: str, pos: str) -> str:
    result = client.translate(lemma, target_language="en")
    print(result)
    return {
        'word': lemma,
        'definitions': {
            pos: lemma + " -> " + result["translatedText"]
        },
        'html': f'<span class="translation">{result["translatedText"]}</span>'
    }

if __name__ == "__main__":
    starttime = time.time()
    result = translate("Super Kuhl!", "ADJ")
    endtime = time.time()
    print(f"Translation took {endtime - starttime:.2f} seconds")
    print(f"Result: {result}")
    print(f"Values: {result['definitions'].values()}")
    for pos, defs in result['definitions'].items():
        print(f"POS: {pos}, Definitions: {defs}")
