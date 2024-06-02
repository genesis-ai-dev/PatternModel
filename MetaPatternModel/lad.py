"""lad.py
Linguistic Anomaly Detection
"""
from difflib import SequenceMatcher

class LinguisticAnomalyDetector:
    """
    Class for performing Linguistic Anomaly Detection (LAD) on texts.
    """
    def __init__(self, source_reference, target_reference):
        self.source_references = source_reference
        self.target_references = target_reference
        assert len(self.source_references) == len(self.target_references), "The reference texts must have the same number of lines."
        self.scores = []

    def _rank_similar(self, references, query):
        """
        Ranks the given references by similarity to the query and returns the indexes of the original references in the order of their similarity.

        Parameters:
        references (list): The list of references to compare against the query.
        query (str): The query string to compare against the references.

        Returns:
        list: A list of indexes representing the original references ranked by similarity to the query.
        """
        # Use SequenceMatcher to compare the query with each reference and store the index with the score
        similarity_scores = [
            (index, SequenceMatcher(None, query, reference).ratio()) 
            for index, reference in enumerate(references)
        ]

        # Sort the indexes by their similarity score in descending order
        sorted_indexes = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Return only the sorted indexes without the scores
        return [index for index, score in sorted_indexes]

    def rank_similar_source(self, query):
        """
        Ranks the source references by similarity to the query.

        Parameters:
        query (str): The query string to compare against the source references.

        Returns:
        list: A list of source references ranked by similarity to the query.
        """
        return self._rank_similar(self.source_references, query)
    
    def rank_similar_target(self, query):
        """
        Ranks the target references by similarity to the query.

        Parameters:
        query (str): The query string to compare against the target references.

        Returns:
        list: A list of target references ranked by similarity to the query.
        """
        return self._rank_similar(self.target_references, query)

    def detect(self, target_draft, source_baseline):
        """
        Detects the similarity between the target draft and the source baseline by ranking them and comparing the rankings.

        Parameters:
        target_draft (str): The draft text of the target document to be compared.
        source_baseline (str): The baseline text of the source document to be compared.

        Returns:
        float: The similarity score between the target draft and the source baseline as a percentage.
        """
        target_rankings = self.rank_similar_target(target_draft)
        source_rankings = self.rank_similar_source(source_baseline)

        score = SequenceMatcher(None, str(target_rankings), str(source_rankings)).ratio() * 100
        self.scores.append(score)

        return score
    
    def __str__(self):
        return f"Average score: {sum(self.scores)/len(self.scores)} over {len(self.scores)} detections."




