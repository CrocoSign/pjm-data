import json
import re
from interfaces import ValidationResult, Word, WordCorrector
import logging
import os
from contextlib import suppress
from typing import Optional
import torch
from torch import Tensor

from gemini_rate_limit import GeminiAPIManager


class BielikCorrector(WordCorrector):
    def __init__(
        self,
        model_name: str = "speakleash/Bielik-11B-v2.6-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        self._validator = None

    @property
    def validator(self) -> "GeminiValidator":
        if self._validator is None:
            self._validator = GeminiValidator()
        return self._validator

    def _load_model(self):
        if not self._model_loaded:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=torch.bfloat16
            )
            self.model = self.model.to(self.device)  # type: ignore
            self._model_loaded = True

    def correct(self, word: Word, rejected: bool = False) -> str:
        self._load_model()
        self.logger.debug(f"LLM correcting '{word.text}' in: '{word.sentence}'")

        try:
            system_prompt = (
                "Jesteś ekspertem od korekty polskich tekstów dla transkrybcji Polskiego Języka Migowego. "
                "Poprawiasz wyłącznie oczywiste błędy ortograficzne, literówki i powtórzenia liter. "
                "Nie zmieniasz gramatyki, składni ani interpunkcji, teksty w języku migowym nie zawsze są idealną polszczyzną."
                "Jeśli jest literowane słowo, nie poprawiaj go."
            )

            context_parts = []
            if word.previous_sentence:
                context_parts.append(f'Poprzednie zdanie: "{word.previous_sentence}"')

            context_parts.append(f'Obecne zdanie: "{word.sentence}"')

            if word.next_sentence:
                context_parts.append(f'Następne zdanie: "{word.next_sentence}"')

            context = "\n".join(context_parts)

            user_prompt = f"""{context}

                Słowo do sprawdzenia: "{word.text}"

                Jeśli słowo zawiera błąd (literówka, powtórzenia liter, zlepione słowa), zwróć poprawioną wersję.
                Jeśli słowo jest poprawne, zwróć je bez zmian.
                Wykorzystaj kontekst poprzedniego i następnego zdania do lepszego zrozumienia znaczenia.
                Zwróć TYLKO poprawione słowo, bez komentarzy."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            formatted_input: str = self.tokenizer.apply_chat_template(  # type: ignore
                messages, tokenize=False, add_generation_prompt=True
            )

            input_ids: Tensor = self.tokenizer.encode(  # type: ignore
                formatted_input, return_tensors="pt"
            )
            model_inputs = input_ids.to(self.device)

            generated_ids = self.model.generate(  # type: ignore
                model_inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id,  # type: ignore
            )

            corrected: str = self.tokenizer.decode(  # type: ignore
                generated_ids[0][len(input_ids[0]) :], skip_special_tokens=True
            ).strip()

            corrected: str = corrected.strip('"').strip("'").strip()
            if corrected == word.text:
                self.logger.info(f"LLM correction unchanged: '{word.text}'")
                return word.text

            validation_result = self.validator(word, corrected)
            if validation_result.is_valid:
                self.logger.info(
                    f"LLM correction accepted: '{word.text}' -> '{corrected}'"
                )
                return corrected
            elif not rejected:
                self.logger.info(
                    f"LLM correction rejected by validator: '{word.text}' -> '{corrected}', retrying without correction"
                )
                second_correction = self.correct(word, rejected=True)
                return second_correction
            else:
                self.logger.info(
                    f"LLM correction rejected by validator: '{word.text}' -> '{corrected}'"
                )
                return word.text

        except Exception as e:
            self.logger.error(f"LLM correction failed: {e}")
            return word.text

    def cleanup(self):
        if self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self._validator:
            self._validator.close()
            self._validator = None

        self.model = None
        self.tokenizer = None
        self._model_loaded = False

def get_keys() -> list[str]:
    keys = os.getenv("GEMINI_API_KEYS")
    if keys:
        raw_keys = keys.split(",")
    else:
        raise ValueError("GEMINI_API_KEYS not provided and not found in environment")
    return raw_keys

class GeminiValidator:
    def __init__(self, api_keys: Optional[list[str]] = None) -> None:
        if not api_keys:
            api_keys = get_keys()
        
        self.api_keys = api_keys
        self.api_manager = GeminiAPIManager(self.api_keys, model_name="gemini-2.5-flash")
        self.logger = logging.getLogger(self.__class__.__name__)

    def __call__(
        self, word: Word, candidate: str, unknown_retry: bool = False
    ) -> ValidationResult:
        try:
            self.logger.debug(f"Validating correction: '{word.text}' -> '{candidate}'")
            system_prompt = """
                <Role>
                You are a validator for Polish word corrections.
                Given an original word and a candidate correction, decide whether the candidate is a valid correction.
                </Role>
                <Rules>
                - Accept if the candidate is a correct Polish word or a reasonable fix (e.g., typo, missing letter).
                - Reject if the candidate introduces unrelated English text, meta prompts, or symbols.
                - Return strictly in JSON.
                </Rules>
                <Example>
                <Input>
                {"original": "zzdrowie", "candidate": "zdrowie", "sentence": "Dbaj o zzdrowie"}
                </Input>
                <Output>
                {"decision": "ACCEPT", "reason": "Corrected duplicated letter"}
                </Output>
                </Example>
                Now validate:
                """

            user_prompt = f"""{system_prompt}
                Original: "{word.text}"
                Candidate: "{candidate}"
                Sentence: "{word.sentence}"
                
                Respond in JSON with "decision" as "ACCEPT" or "REJECT" and a "reason".
                """
            response = self.api_manager.generate_content(user_prompt)
            if "error" in response:
                self.logger.error(f"API error: {response['error']}")
                # raise ValueError(f"API error: {response['error']}")
            text = (
                response.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            text = text.strip()
            self.logger.debug(f"Validator response: {text}")

            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in response")

            result_json = match.group(0)
            result = json.loads(result_json)

            decision = result.get("decision", "").upper()
            reason = result.get("reason", "No reason provided")

            self.logger.debug(f"Validation result: {decision}, {reason}")

            if decision == "ACCEPT":
                return ValidationResult(is_valid=True, reason=reason, confidence=0.95)
            elif decision == "REJECT":
                return ValidationResult(is_valid=False, reason=reason, confidence=0.95)
            else:
                if not unknown_retry:
                    self.logger.warning(f"Unknown decision: {decision}, retrying once")
                    return self.__call__(word, candidate, unknown_retry=True)
                else:
                    self.logger.error(
                        f"Unknown decision after retry: {decision}, rejecting"
                    )
                    return ValidationResult(
                        is_valid=False, reason="Unknown decision", confidence=0.0
                    )

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return ValidationResult(
                is_valid=False, reason="Validation error", confidence=0.0
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if hasattr(self, "client") and self.client:  # type: ignore
            with suppress(Exception):
                self.client.close()  # type: ignore


class GeminiCorrector(WordCorrector):
    def __init__(self, api_keys: Optional[list[str]] = None) -> None:
        if not api_keys:
            api_keys = get_keys()
        
        self.api_keys = api_keys
        self.client = GeminiAPIManager(self.api_keys, model_name="gemini-2.5-flash")
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validator = None

    @property
    def validator(self) -> "GeminiValidator":
        if self._validator is None:
            self._validator = GeminiValidator()
        return self._validator

    def _build_context(self, word: Word) -> str:
        """Build context string from word information"""
        context_parts = []

        if word.previous_sentence:
            context_parts.append(f'Poprzednie zdanie: "{word.previous_sentence}"')

        context_parts.append(f'Obecne zdanie: "{word.sentence}"')

        if word.next_sentence:
            context_parts.append(f'Następne zdanie: "{word.next_sentence}"')

        return "\n".join(context_parts)

    def correct(self, word: Word, rejected: bool = False) -> str:
        """Correct word using Gemini"""
        self.logger.debug(f"Gemini correcting '{word.text}' in: '{word.sentence}'")

        try:
            system_prompt = (
                "You are an expert in correcting Polish texts for Polish Sign Language (PJM) transcriptions. "
                "You only fix obvious spelling errors, typos, and unintentional letter repetitions. "
                "You also correct whole words written in ALL CAPS when they are not acronyms or proper nouns. "
                "For onomatopoeia, standardize to two repetitions of the base sound (e.g., 'hahahaha' → 'haha', 'brrrrr' → 'brr'). "
                "Do not change grammar, syntax, or punctuation. Sign language transcriptions are not always perfect Polish. "
                "If a word is spelled out letter-by-letter, do not correct it."
            )

            context = self._build_context(word)

            user_prompt = f"""
            {system_prompt}

            Context: {context}

            Word to check: "{word.text}"

            Instructions:
            1. If the word contains a typo, return the corrected version (e.g., 'dzein' → 'dzień').
            2. If the word has unintentional repeated letters, fix it (e.g., 'dooom' → 'dom').
            3. If the word is onomatopoeia, standardize to two repetitions of the base sound:
               - 'hahahaha' → 'haha'
               - 'hehehehe' → 'hehe'
               - 'brrrrrrr' → 'brr'
               - 'psssssss' → 'pss'
            4. If the word is written in ALL CAPS and is not an acronym (like TV, USA) or proper noun, convert to lowercase (e.g., 'DOM' → 'dom', but 'TV' → 'TV').
            5. If words are merged without spaces, separate them (e.g., 'domkot' → 'dom kot').
            6. If the word is correct, return it unchanged.
            7. Use the context of the previous and next sentences to better understand the meaning.
            8. Return ONLY the corrected word, without any comments or explanations.
            9. Preserve any punctuation attached to the word.

            Examples:
            - "dooom" → "dom"
            - "hahahaha" → "haha"
            - "brrrrr" → "brr"
            - "DOM" → "dom"
            - "TV" → "TV"
            - "dzein" → "dzień"
            - "zdrowie!" → "zdrowie!"
            - "PJM" → "PJM"
        
            Return format: [CORRECTED_WORD]
            """
            response = self.client.generate_content(user_prompt)
            if "error" in response:
                self.logger.error(f"API error: {response.get('error')}")
                return word.text

            corrected = response["candidates"][0]["content"]["parts"][0]["text"].strip()
            corrected = corrected.strip('"').strip("'").strip("[]").strip()

            self.logger.info(f"Gemini response: {response}")
            if corrected == word.text:
                self.logger.info(f"LLM correction unchanged: '{word.text}'")
                return word.text

            validation_result = self.validator(word, corrected)
            if validation_result.is_valid:
                self.logger.info(
                    f"LLM correction accepted: '{word.text}' -> '{corrected}'"
                )
                return corrected
            elif not rejected:
                self.logger.info(
                    f"LLM correction rejected by validator: '{word.text}' -> '{corrected}', retrying without correction"
                )
                second_correction = self.correct(word, rejected=True)
                return second_correction
            else:
                self.logger.info(
                    f"LLM correction rejected by validator: '{word.text}' -> '{corrected}'"
                )
                return word.text

        except Exception as e:
            self.logger.error(f"Gemini correction failed: {e}")
            return word.text

    def __del__(self):
        if hasattr(self, "client"):
            with suppress(Exception):
                self.client.close()  # type: ignore
