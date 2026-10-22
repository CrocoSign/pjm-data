from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CaptionId:
    filename: str
    caption_id: str

    def to_dict(self) -> dict:
        return {"filename": {self.filename}, "caption_id": {self.caption_id}}


@dataclass
class LemmaStats:
    lemma: str
    count: int
    caption_ids: list[CaptionId] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "lemma": self.lemma,
            "count": self.count,
            "caption_ids": self.caption_ids,
        }


@dataclass
class Word:
    text: str
    position: int
    sentence: str
    previous_sentence: str = ""
    next_sentence: str = ""


@dataclass
class ValidationResult:
    is_valid: bool
    reason: Optional[str] = None
    confidence: float = 1.0


class LemmaExtractor:
    def extract_lemma(self, text: str) -> list[str]:
        raise NotImplementedError


class TextProcessor:
    def process(self, path: str | Path) -> str:
        raise NotImplementedError


class DataRepository:
    def save(self, data: dict, path: str | Path) -> None:
        raise NotImplementedError

    def load(self, path: str | Path) -> dict:
        raise NotImplementedError


class WordValidator:
    def validate(self, word: Word) -> ValidationResult:
        raise NotImplementedError


class WordCorrector:
    def correct(self, word: Word) -> str:
        raise NotImplementedError


class CaptionProcessor:
    def process(self, text: str, previous_sentence: str, next_sentence: str) -> str:
        raise NotImplementedError


class ProcessingPipeline:
    def process_file(self, *args, **kwargs):
        raise NotImplementedError

    def process_directory(self, *args, **kwargs):
        raise NotImplementedError
