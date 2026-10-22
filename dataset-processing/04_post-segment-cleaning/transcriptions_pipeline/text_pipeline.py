import logging
from pathlib import Path
import re
import time
from collections import Counter

from interfaces import (
    CaptionProcessor,
    DataRepository,
    LemmaExtractor,
    LemmaStats,
    ProcessingPipeline,
    TextProcessor,
    Word,
    WordCorrector,
    WordValidator,
)
from validators import CompositeValidator, HunspellValidator, RegexValidator
from correctors import BielikCorrector, GeminiCorrector  # noqa: F401
import webvtt
import json
import morfeusz2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class VTTTextProcessor(TextProcessor):
    def process(self, path: str | Path) -> str:
        if isinstance(path, Path):
            path = str(path)
        vtt = webvtt.read(path)
        text = "".join([caption.text for caption in vtt.captions])
        text = re.sub(r"[^\w\s]", " ", text)
        return text.lower()


class MorfeuszLemmaExtractor(LemmaExtractor):
    def __init__(self):
        self.morfeusz = morfeusz2.Morfeusz()

    def extract_lemma(self, text: str) -> list[str]:
        words = text.split()
        lemmas = []

        for word in words:
            analysis = self.morfeusz.analyse(word)
            if analysis:
                lemma = analysis[0][2][1]
                lemma = lemma.split(":")[0]
                lemmas.append(lemma)

        return lemmas


class JSONRepository(DataRepository):
    def save(self, data: dict, path: str | Path) -> None:
        if isinstance(path, str):
            path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=5)

    def load(self, path: str | Path) -> dict:
        if isinstance(path, str):
            path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


class SmartCaptionProcessor(CaptionProcessor):
    def __init__(self, validator: WordValidator, corrector: WordCorrector) -> None:
        self.validator = validator
        self.corrector = corrector
        self.logger = logging.getLogger(self.__class__.__name__)

    def _extract_words(
        self, text: str, previous_sentence: str, next_sentence: str
    ) -> list[Word]:
        words: list[Word] = []
        raw_words: list[str] = text.split()
        for i, raw_word in enumerate(raw_words):
            words.append(
                Word(
                    text=raw_word,
                    position=i,
                    sentence=text,
                    previous_sentence=previous_sentence,
                    next_sentence=next_sentence,
                )
            )
        return words

    def process(self, text: str, previous_sentence: str, next_sentence: str) -> str:
        words: list[Word] = self._extract_words(
            text=text, previous_sentence=previous_sentence, next_sentence=next_sentence
        )
        corrected_words: list[Word] = []
        correction_made: int = 0

        for word in words:
            validation = self.validator.validate(word)
            if not validation.is_valid:
                logging.debug(
                    f"Invalid word was detected: {word.text} ({validation.reason})"
                )
                corrected = self.corrector.correct(word)

                if corrected != word.text:
                    correction_made += 1
                    corrected_word = Word(
                        text=corrected,
                        position=word.position,
                        sentence=word.sentence,
                        previous_sentence=word.previous_sentence,
                        next_sentence=word.next_sentence,
                    )
                    corrected_words.append(corrected_word)
                    self.logger.debug(
                        f"Corrected '{word.text}' to '{corrected}' in sentence '{word.sentence}'."
                    )
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        if correction_made > 0:
            self.logger.info(f"Made {correction_made} corrections in caption")
        return " ".join([w.text for w in corrected_words])


class VTTFixer(ProcessingPipeline):
    def __init__(self, caption_processor: CaptionProcessor) -> None:
        self.caption_processor = caption_processor
        self.logger = logging.getLogger(self.__class__.__name__)

    def process_file(self, input_path: str | Path, output_path: str | Path) -> int:
        if isinstance(input_path, str):
            input_path = Path(input_path)
        if isinstance(output_path, str):
            output_path = Path(output_path)
        try:
            vtt = webvtt.read(str(input_path), encoding="utf-8")
        except Exception as e:
            vtt = webvtt.read(str(input_path))
            self.logger.warning(f"Failed to read {input_path} as utf-8: {e}")

        correction_count = 0

        for i, caption in enumerate(vtt.captions):
            original_text = caption.text
            normalized_text = self._normalize_unicode(original_text)
            previous_sentence = ""
            next_sentence = ""

            if i > 0:
                previous_sentence = self._normalize_unicode(vtt.captions[i - 1].text)

            if i < len(vtt.captions) - 1:
                next_sentence = self._normalize_unicode(vtt.captions[i + 1].text)
            corrected_text = self.caption_processor.process(
                normalized_text, previous_sentence, next_sentence
            )
            if corrected_text != original_text:
                correction_count += 1
                caption.text = corrected_text

        try:
            with open(output_path, "w", encoding="utf-8", errors="replace") as f:
                f.write("WEBVTT\n\n")
                for caption in vtt.captions:
                    f.write(f"{caption.start} --> {caption.end}\n")
                    f.write(f"{caption.text}\n\n")
            self.logger.debug(f"Saved with UTF-8 encoding: {output_path}")
        except Exception as e:
            self.logger.warning(f"UTF-8 save failed, using fallback: {e}")
            vtt.save(str(output_path))
        self.logger.info(
            f"Processed {input_path.name}: {correction_count} captions corrected"
        )
        return correction_count

    def process_directory(
        self, input_dir: str | Path, output_dir: str | Path
    ) -> dict[str, int]:
        self.logger.debug(f"Fixing VTT files in {input_dir} and saving to {output_dir}")
        if isinstance(input_dir, str):
            input_dir = Path(input_dir)
        if output_dir is None:
            self.logger.warning("No output_dir provided to process_directory.")
            return {}
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_files = list(input_dir.glob("*.vtt"))
        if not input_files:
            self.logger.warning(f"No VTT files found in {input_dir}")
            return {}

        results = {}
        for vtt_file in input_files:
            self.logger.info(f"Processing file: {vtt_file.name}")
            try:
                output_path = output_dir / vtt_file.name
                corrections = self.process_file(vtt_file, output_path)
                results[vtt_file.name] = corrections
                self.logger.info(f"Saved fixed file to: {output_path}")
            except Exception as e:
                self.logger.error(f"Failed to process {vtt_file.name}: {e}")
                results[vtt_file.name] = -1
                continue
        return results

    def _normalize_unicode(self, text: str) -> str:
        replacements = {
            "\u2018": "'",  # Left single quotation mark
            "\u2019": "'",  # Right single quotation mark
            "\u201c": '"',  # Left double quotation mark
            "\u201d": '"',  # Right double quotation mark
            "\u201e": '"',  # Double low-9 quotation mark
            "\u2026": "...",  # Horizontal ellipsis
            "\u2013": "-",  # En dash
            "\u2014": "--",  # Em dash
        }

        for unicode_char, ascii_char in replacements.items():
            text = text.replace(unicode_char, ascii_char)

        return text


class WordBank(ProcessingPipeline):
    def __init__(
        self,
        text_processor: TextProcessor,
        lemma_extractor: LemmaExtractor,
        data_repository: DataRepository,
        fixer: VTTFixer,
    ):
        self.text_processor = text_processor
        self.lemma_extractor = lemma_extractor
        self.data_repository = data_repository
        self.fixer = fixer

    def process_file(self, path: str | Path) -> list[str]:
        text = self.text_processor.process(path=path)
        lemmas = self.lemma_extractor.extract_lemma(text=text)
        return lemmas

    def process_directory(self, path: str | Path) -> dict[str, list[str]]:
        results = {}

        if isinstance(path, str):
            path = Path(path)

        for file in path.glob("*.vtt"):
            file_id = file.stem
            lemmas = self.process_file(file)
            results[file_id] = lemmas

        return results

    def build_statistics(
        self, lemmas_by_file: dict[str, list[str]]
    ) -> list[LemmaStats]:
        lemmas_to_file = {}
        lemma_counts = Counter()

        for file_id, lemmas in lemmas_by_file.items():
            lemma_counts.update(lemmas)
            for lemma in set(lemmas):
                if lemma not in lemmas_to_file:
                    lemmas_to_file[lemma] = []
                lemmas_to_file[lemma].append(file_id)

        stats = [
            LemmaStats(lemma=lemma, count=count, caption_ids=lemmas_to_file[lemma])
            for lemma, count in lemma_counts.items()
        ]

        return sorted(stats, key=lambda x: x.count, reverse=True)

    def fix(self, input_dir: str | Path, output_dir: str | Path) -> None:
        self.fixer.process_directory(input_dir=input_dir, output_dir=output_dir)
        corrector = getattr(self.fixer.caption_processor, "corrector", None)
        if corrector and hasattr(corrector, "cleanup"):
            corrector.cleanup()

    def save_statistics(self, stats: list[LemmaStats], output_path: str | Path) -> None:
        data = {
            "total_unique_lemmas": len(stats),
            "lemmas": [s.to_dict() for s in stats],
        }
        self.data_repository.save(data, output_path)


def main():
    logging.getLogger()

    base_dir = Path(__file__).resolve().parent.parent.parent
    transcription_dir = base_dir / "data" / "vtts"
    fixed_transcription_dir = base_dir / "data" / "vtts_fixed"
    vtt_files = list(transcription_dir.glob("*.vtt"))

    logging.info("Setting up processing pipeline...")
    validators = [RegexValidator(), HunspellValidator()]
    composite_validator = CompositeValidator(validators=validators)
    corrector = GeminiCorrector()
    caption_processor = SmartCaptionProcessor(
        validator=composite_validator, corrector=corrector
    )

    word_bank = WordBank(
        text_processor=VTTTextProcessor(),
        lemma_extractor=MorfeuszLemmaExtractor(),
        data_repository=JSONRepository(),
        fixer=VTTFixer(caption_processor=caption_processor),
    )

    if not vtt_files:
        logging.warning(f"No VTT files found in {transcription_dir}")
        logging.info(f"Directory contents: {list(transcription_dir.iterdir())}")
        exit(1)

    word_bank.fixer.process_directory(
        input_dir=transcription_dir, output_dir=fixed_transcription_dir
    )
    logging.info(
        f"Found {len(vtt_files)} VTT files: {[f.name for f in vtt_files[:5]]}..."
    )

    output_file = Path("lemma_statistics.json")

    logging.info("Processing files...")
    lemmas_by_file = word_bank.process_directory(fixed_transcription_dir)

    logging.info(f"Processed {len(lemmas_by_file)} files")
    logging.info(
        "Sample results:", {k: len(v) for k, v in list(lemmas_by_file.items())[:3]}
    )

    logging.info("Building statistics...")
    stats = word_bank.build_statistics(lemmas_by_file)

    logging.info(f"Found {len(stats)} unique lemmas")
    logging.info("Top 10 most frequent lemmas:")
    for s in stats[:10]:
        logging.info(
            f"  {s.lemma}: {s.count} occurrences in {len(s.caption_ids)} files"
        )

    logging.info(f"\nSaving to {output_file}...")
    word_bank.save_statistics(stats, output_file)


if __name__ == "__main__":
    logging.basicConfig(
        filename=f".logs/transcription_pipeline{int(time.time())}.log",
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
