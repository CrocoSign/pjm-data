import json
import logging
import os
import re

import deepl
from deepl import TextResult
from dotenv import load_dotenv
from tqdm import tqdm

from hdf5_manager import HDF5Loader, HDF5Saver

logger = logging.getLogger(__name__)


class Translator:
    def __init__(
        self,
        source_file: str,
        destination_file: str,
        source_lang: str = "PL",
        target_lang: str = "EN-US",
        batch_size: int = 50,
    ):
        load_dotenv()
        api_key: str | None = os.getenv("DeepL_API_Key")
        if not api_key:
            raise ValueError("DeepL_API_Key not found in env")

        self.translator = deepl.Translator(api_key)
        self.source_file: str = source_file
        self.destination_file: str = destination_file
        self.source_lang: str = source_lang
        self.target_lang: str = target_lang
        self.batch_size: int = batch_size

        self.source: dict = {}
        self.translated: dict = {}

    def load(self):
        with HDF5Loader(self.source_file) as loader:
            self.source = loader.load()

        if os.path.exists(self.destination_file):
            with HDF5Loader(self.destination_file) as loader:
                self.translated = loader.load()

    def flag_existing(self, flag: str = "deepl"):
        with HDF5Saver(self.destination_file) as saver:
            for key, value in self.translated.items():
                saver.save({key: value}, flag=flag)
        logger.info(f"Flagged {len(self.translated)} existing translations as '{flag}'")

    def _get_pending(self) -> dict:
        return {
            k: v
            for k, v in self.source.items()
            if k not in self.translated or self.translated[k] == v
        }

    def translate(self):
        self.load()
        to_translate: dict = self._get_pending()

        char_count: int = sum(len(v) for v in to_translate.values())
        logger.info(
            f"Estimated characters to translate: {char_count:,} (free tier: 500,000/month)"
        )

        total: int = len(self.source)
        remaining: int = len(to_translate)
        logger.info(
            f"Total: {total}, Already done: {total - remaining}, Remaining: {remaining}"
        )

        if not to_translate:
            logger.info("Nothing to translate!")
            return

        keys: list = list(to_translate.keys())
        with HDF5Saver(self.destination_file) as saver:
            for i in tqdm(
                range(0, len(keys), self.batch_size),
                desc="Translating",
                unit="batch",
            ):
                batch_keys: list[str] = keys[i : i + self.batch_size]
                batch_texts: list[str] = [to_translate[k] for k in batch_keys]

                try:
                    results: TextResult | list[TextResult] = (
                        self.translator.translate_text(
                            batch_texts,
                            source_lang=self.source_lang,
                            target_lang=self.target_lang,
                        )
                    )
                except deepl.DeepLException as e:
                    logger.error(
                        f"DeepL API error at batch {i // self.batch_size}: {e}"
                    )
                    raise

                saver.save({k: r.text for k, r in zip(batch_keys, results)}, flag="deepl")

                done: int = min(i + self.batch_size, len(keys))
                logger.info(f"Progress: {done}/{remaining}")
    
    def patch_translations(self, problematic_file: str):
        self.load()
        if not os.path.exists(problematic_file):
            logger.warning(f"Problematic file not found: {problematic_file}")
            return

        with open(problematic_file, "r", encoding="utf-8") as f:
            corrections: dict = json.load(f)

        source_to_keys: dict = {}
        for key, text in self.source.items():
            source_to_keys.setdefault(text, []).append(key)

        patched: int = 0
        with HDF5Saver(self.destination_file) as saver:
            for source_text, correct_translation in corrections.items():
                for key in source_to_keys.get(source_text, []):
                    saver.save({key: correct_translation}, flag="manual")
                    patched += 1

        logger.info(f"Patched {patched} translations from {problematic_file}")

    def find_untranslated(
        self,
        common_words: set,
        problematic_file: str = "",
    ):
        self.load()

        known_problematic: set = set()
        problematic_words: set = set()
        if problematic_file and os.path.exists(problematic_file):
            with open(problematic_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                known_problematic = set(data.keys())
                for k in data.keys():
                    cleaned = re.sub(r'[^\w\s]', '', k.lower()).strip()
                    if ' ' not in cleaned:
                        problematic_words.add(cleaned)

        issues: list = []
        for key in self.translated:
            if key not in self.source or self.translated[key] == self.source[key]:
                continue

            if self.source[key] in known_problematic:
                continue

            src_words: set = {w for w in re.sub(r'[^\w\s]', '', self.source[key].lower()).split() if not w.isdigit()}
            target_words: set = {w for w in re.sub(r'[^\w\s]', '', self.translated[key].lower()).split() if not w.isdigit()}

            if not src_words:
                continue

            overlap: set = (src_words & target_words) - common_words - problematic_words
            ratio: float = len(overlap) / len(src_words)

            if ratio > 0:
                issues.append((key, ratio, self.source[key], self.translated[key]))

        issues.sort(key=lambda x: x[1], reverse=True)

        with HDF5Saver(self.destination_file) as saver:
            for key, ratio, src, target in issues:
                saver.save({key: target}, flag="invalid")

        logger.info(f"Suspected untranslated entries: {len(issues)}")
        for key, ratio, src, target in issues:
            logger.info(f"\n  [{key}] overlap: {ratio:.0%}")
            logger.info(f"    {self.source_lang}: {src}")
            logger.info(f"    {self.target_lang}: {target}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="untranslated.log",
        filemode="w",
    )
    t = Translator(
        source_file="./data/texts.h5",
        destination_file="./data/texts_eng.h5",
    )
    t.load()

    # 1. Flag all existing translations as "deepl"
    t.flag_existing()

    # 2. Apply manual corrections from problematic.json
    t.patch_translations("./problematic.json")

    # 3. Find untranslated and flag as "invalid"
    t.find_untranslated(
        problematic_file="./problematic.json",
        common_words={
        # Interjections / sounds
        "aha", "ah", "oh", "ooh", "oo", "aa", "aaa", "eee", "ha", "haha", "hm", "hmm",
        "eh", "ech", "pff", "pfff", "bam", "boom", "psst", "shh", "wow", "bravo",
        "halo", "oops", "ooo", "mmm", "ehh", "yy", "oj", "e", "mhm",
        # Short common words valid in both languages
        "a", "o", "ok", "we", "so", "op", "by",
        "super", "stop", "top", "zero", "weekend",
        "plus", "moment", "region", "same",
        # International / loanwords
        "bar", "bus", "pub", "club", "film", "sport", "park", "bank", "plan", "fast", "food",
        "menu", "test", "team", "fan", "grill", "trend", "boom", "limit",
        "hotel", "motel", "hostel", "radio", "taxi", "metro",
        "pizza", "pasta", "sushi", "kebab", "burger",
        "rock", "jazz", "pop", "rap", "blues", "punk", "disco", "hip-hop",
        "golf", "tenis", "squash", "judo", "karate", "yoga",
        "system", "post", "model", "auto", "metal", "mini", "parking",
        "sanatorium", "darts", "monitor", "instrument", "neon",
        "aquapark", "hetero", "ufo", "sms", "hobby",
        # Tech / media
        "internet", "email", "laptop", "komputer", "telefon", "smartfon",
        "tv", "dvd", "cd", "gps", "wifi", "bluetooth", "usb", "tablet",
        "software", "hardware", "online", "offline",
        # Units / formats
        "km", "kg", "ml", "cm", "mm", "mg",
        # Names / titles often left as-is
        "charlie", "chaplin", "maria", "micek", "donald",
        "przemyśl", "olsztyn", "zorro", "miki", "mike", "grand", "filip",
        "morskie", "oko", "mc", "donalds", "chuptio", "pepsi",
        "apple", "mickey", "atom", "mazur", "mzk", "cybulski", "pzg",
        "obi", "leroy", "merlin", "kasi", "disney", "flap", "flip", "chalin",
        "oovoo", "moka", "rewa", "ania", "dobromir", "ariadna",
        "pko", "bhp", "pomot", "pfron", "uw", "skm", "wsk", "sjm", "pjm", "zwm", "gk",
        "kfc", "mcdonalds", "skype", "wars", "american", "pie", "pro",
        "xxl", "hut", "nogs", "spit", "islam", "tshirt",
        # Food
        "hamburger", "banana",
        # Other loanwords
        "bowling", "alarm", "gong", "problem", "sauna", "bikini", "wagon",
        "minus", "symbol", "humor", "bomba",
        # Ignored — Polish/English overlap but different meaning
        "no", "to", "do", "on", "go", "pan", "pani", "i", "not",
        # Single letters (initials, abbreviations)
        "l", "m", "p",
    })
