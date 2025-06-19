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

def translate(lemma: str, pos: str) -> str:
    result = argostranslate.translate.translate(lemma, "de", "en")
    return {
        'word': lemma,
        'definitions': {
            pos: lemma + " -> " + result
        },
        'html': f'<span class="translation">{result}</span>'
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
