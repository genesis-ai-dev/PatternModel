from models import ClaudeMetaTranslator, GPT4MetaTranslator, extract_numbers
from lad import LinguisticAnomalyDetector
from random import choices
import numpy as np
import tqdm


class Translator:
    """
    handles translation of much more text, also uses LAD
    """
    def __init__(self, aligned_texts: tuple[np.ndarray, np.ndarray], prompts: list[str], claude_key: str, lad_level: int = 10):
        self.aligned_texts = aligned_texts
        self.prompts = prompts
        self.claude_key = claude_key
        choice = choices(list(range(len(aligned_texts[0]))), k=lad_level)
        self.linguistic_anomaly_detector = LinguisticAnomalyDetector(source_reference=self.aligned_texts[0][choice], target_reference=self.aligned_texts[1][choice])
    
    def translate(self) -> list[str]:
        """
        Starts the translation loop
        """
        translations = []
        for prompt in self.prompts:
            examples = self.search(prompt)
            translator = ClaudeMetaTranslator(self.claude_key)
            translation = translator.translate(prompt, pairs=examples)

            sequence = extract_numbers(translation)
            tokens = [translator.tokenizer.value_to_token.get(int(num), '[unclear]') for num in sequence]
            text = " ".join(tokens)
            translations.append(text)
            score = self.linguistic_anomaly_detector.detect(text, prompt)
            print("Translation: ")
            print("Score: " + str(score))
            print("Text: " + text)
            print("\n\n\n")
            
        return translations


    def search(self, query) -> str:
        """
        Searches for most relavent examples. TODO
        """
        return query