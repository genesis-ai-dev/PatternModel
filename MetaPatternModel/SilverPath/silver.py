"""
Where a "Golden Path" tells which examples one needs to curate in order for an optimal dataset, a silver path takes whatever examples you may already have and determines what will most easily be translated given what you already have. 
This can help provide a metric for how confident one should be about a generated translation, 
as well as a means of determining which materials might be highest priority. 
"""
import os
import json
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import tqdm

class SilverPath:
    def __init__(self, data_dir: str, max_rank: int = 100):
        self.data_dir = data_dir
        self.source_texts = self.load_text_file("source.txt")
        self.target_texts = self.load_text_file("target.txt")
        self.prompts = self.load_text_file("prompts.txt")
        self.max_rank = max_rank
        
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(self.source_texts)

    def load_text_file(self, file_name: str) -> list[str]:
        file_path = os.path.join(self.data_dir, file_name)
        with open(file_path, "r") as file:
            return [line.strip().lower().translate(str.maketrans('', '', string.punctuation)) for line in file]
    
    def search(self, query: str) -> int:
        query_words = set(query.split())
        rank = 0
        previous_similar_words = None
        while query_words and rank < self.max_rank:
            query_vector = self.vectorizer.transform([' '.join(query_words)])
            cosine_similarities = np.dot(query_vector, self.vectorizer.transform(self.source_texts).T).toarray().flatten()            
            most_similar_index = np.argmax(cosine_similarities)
            similar_words = set(self.source_texts[most_similar_index].split())
            if similar_words == previous_similar_words:
                rank *= len(similar_words)
                break
            previous_similar_words = similar_words
            query_words -= similar_words
            rank += 1
        return rank

    def rank_prompts(self) -> None:
        output_file = os.path.join(self.data_dir, "ranked_prompts.jsonl")
        with open(output_file, "w") as file:
            for prompt in tqdm.tqdm(self.prompts):
                rank = self.search(prompt)
                data = {"prompt": prompt, "rank": rank, "index": self.prompts.index(prompt)}
                file.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    # Usage
    data_dir = "/Users/daniellosey/Desktop/code/biblica/pattern model/languages"
    silver_path = SilverPath(data_dir)
    silver_path.rank_prompts()