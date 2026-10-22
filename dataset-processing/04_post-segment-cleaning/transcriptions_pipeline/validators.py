import logging
import re
from collections.abc import Sequence

from interfaces import ValidationResult, Word, WordValidator


class RegexValidator(WordValidator):
    def __init__(self):
        self.patterns = {
            "repetitions": r"([A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż])\1{2,}",
            "mixed_case": r"[a-ząćęłńóśżź][A-ZĄĆĘŁŃÓŚŻŹ]",
            "caps_case": r"[A-ZĄĆĘŁŃÓŚŻŹ]{2,}",
            "too_long": r".{21,}",
        }
        self.logger = logging.getLogger(self.__class__.__name__)

    def validate(self, word: Word) -> ValidationResult:
        if word.text.strip() in ["...", "…"]:
            return ValidationResult(is_valid=True, reason="Ellipsis", confidence=1.0)

        for pattern_name, pattern in self.patterns.items():
            if re.search(pattern=pattern, string=word.text):
                self.logger.debug(
                    f"Word '{word.text}' failed validation: {pattern_name}"
                )
                return ValidationResult(
                    is_valid=False, reason=pattern_name, confidence=0.9
                )
        return ValidationResult(is_valid=True, reason="RegexValidator", confidence=1.0)


class HunspellValidator(WordValidator):
    def __init__(self, dict_path: str = "pl_PL") -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        try:
            import hunspell

            self.speller = hunspell.HunSpell(
                f"/usr/share/hunspell/{dict_path}.dic",
                f"/usr/share/hunspell/{dict_path}.aff",
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Hunspell: {e}")
            self.speller = None

    def validate(self, word):
        word = Word(
            text=word.text.strip(".,!?;:\"'()[]{}…—-“..."),
            position=word.position,
            sentence=word.sentence,
        )
        if self.speller is None:
            return ValidationResult(
                is_valid=False, reason="Hunspell not initialized", confidence=0.0
            )
        if not word.text:
            return ValidationResult(
                is_valid=True, reason="Punctuation only", confidence=1.0
            )

        if self.speller.spell(word.text):
            return ValidationResult(is_valid=True, reason="Hunspell", confidence=1.0)
        else:
            self.logger.debug(f"Word '{word.text}' not found in Hunspell dictionary")
            return ValidationResult(
                is_valid=False, reason="Not in Hunspell dictionary", confidence=1.0
            )


class CompositeValidator(WordValidator):
    def __init__(self, validators: Sequence[WordValidator]) -> None:
        self.validators = validators
        self.logger = logging.getLogger(self.__class__.__name__)

    def validate(self, word: Word) -> ValidationResult:
        for validator in self.validators:
            result = validator.validate(word)
            if not result.is_valid:
                self.logger.debug(
                    f"Word '{word.text}' failed validation by {validator.__class__.__name__}: {result.reason}"
                )
                return result
        return ValidationResult(
            is_valid=True, reason="All validators passed", confidence=1.0
        )
