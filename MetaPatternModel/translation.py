"""translation.py
Automatic translation.
"""
import os
from models import ClaudeMetaTranslator, GPT4MetaTranslator, extract_numbers
from lad import LinguisticAnomalyDetector
from random import choices
import numpy as np
import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import string

class Translator:
    def __init__(self, data_dir: str, claude_key: str, lad_level: int = 20):
        self.data_dir = data_dir
        self.source_texts = self.load_text_file("source.txt")
        self.target_texts = self.load_text_file("target.txt")
        assert len(self.source_texts) == len(self.target_texts), "The texts must have the same lengths"

        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(self.source_texts)
        
        self.prompts = self.load_text_file("prompts.txt")
        self.claude_key = claude_key
        lad_indices = choices(range(len(self.source_texts)), k=lad_level)
        self.linguistic_anomaly_detector = LinguisticAnomalyDetector(source_reference=[self.source_texts[i] for i in lad_indices],
                                                                     target_reference=[self.target_texts[i] for i in lad_indices])
    
    def load_text_file(self, file_name: str) -> list[str]:
        file_path = os.path.join(self.data_dir, file_name)
        with open(file_path, "r") as file:
            return [line.strip().lower().translate(str.maketrans('', '', string.punctuation)) for line in file]
    
    def translate(self, resume_from: int = 0) -> None:
        output_file = os.path.join(self.data_dir, "output.txt")
        with open(output_file, "a") as file:
            for i in tqdm.tqdm(range(resume_from, len(self.prompts))):
                prompt = self.prompts[i]
                examples = self.search(prompt)
                translator = ClaudeMetaTranslator(self.claude_key)
                translation = translator.translate(prompt, pairs=examples)

                sequence = extract_numbers(translation)
                tokens = [translator.tokenizer.value_to_token.get(int(num), '[unclear]') for num in sequence]
                text = " ".join(tokens)
                score = self.linguistic_anomaly_detector.detect(text, prompt)
                file.write(f"{prompt} -> {text} (LAD Score: {score})\n")

    def search(self, query: str, n: int = 20) -> str:
        results = ""
        for _ in range(n):
            query_vector = self.vectorizer.transform([query])
            cosine_similarities = np.dot(query_vector, self.vectorizer.transform(self.source_texts).T).toarray().flatten()            
            most_similar_indices = np.argpartition(cosine_similarities, -2)[-1:]
            for index in most_similar_indices:
                source_text = self.source_texts[index]
                target_text = self.target_texts[index]
                results += f"source: {source_text}\ntarget: {target_text}\n"

            query_words = set(query.split())
            for index in most_similar_indices:
                similar_words = set(self.source_texts[index].split())
                query_words -= similar_words

            query = ' '.join(query_words)
            if not query:
                break
        return results.strip()


def main():
    data_dir = "/Users/daniellosey/Desktop/code/biblica/pattern model/languages"  # Directory containing source.txt, target.txt, and prompts.txt
    translator = Translator(data_dir, claude_key)
    
    resume_from = 0  # Index of the prompt to resume from (0 to start from the beginning)
    translator.translate(resume_from)


if __name__ == "__main__":
    main()