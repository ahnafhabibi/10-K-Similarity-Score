import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


class SimilarityScoreAlgo:
    """
    A class for computing similarity scores between two large documents.
    The two sentences are the list of sentences in the two respective documents.
    We use an sentence transformer model to achieve this. 

    Example usage:
        sentences1 = ["I love coding.", "Python is awesome."]
        sentences2 = ["Coding is fun.", "I enjoy Python programming."]
        similarity_calculator = SimilarityScoreAlgo("all-MiniLM-L6-v2", sentences1, sentences2)
        embeddings1, embeddings2 = similarity_calculator.encode_sentences()
        similarity_matrix = similarity_calculator.get_similarity_score(embeddings1, embeddings2)
        row_mean, col_mean = similarity_calculator.generate_score(similarity_matrix)
        f1 = similarity_calculator.f1_score(row_mean, col_mean)- this is the harmonic mean
        print(f"F1 score: {f1}")
    """

    def __init__(self, model_type, sentences_1, sentences_2) -> None:
        """
        Initializes the SimilarityScoreAlgo instance with a Sentence Transformer model and two sets of sentences.

        Args:
            model_type (str): The type of Sentence Transformer model to use.
            sentences_1 (list): List of sentences or texts for the first set.
            sentences_2 (list): List of sentences or texts for the second set.
        """
        self.model = SentenceTransformer(model_type)
        self.sentences_1 = sentences_1
        self.sentences_2 = sentences_2

    def encode_sentences(self):
        """
        Encodes both sets of sentences into embeddings using the specified Sentence Transformer model.

        Returns:
            tuple: A tuple containing embeddings for sentences_1 and sentences_2.
        """
        embeddings_1 = self.model.encode(self.sentences_1)
        embeddings_2 = self.model.encode(self.sentences_2)
        return embeddings_1, embeddings_2

    def get_similarity_score(self, embeddings_1, embeddings_2):
        """
        Computes the similarity score matrix between the embeddings of two sets of sentences.

        Args:
            embeddings_1 (list): Embeddings of sentences_1.
            embeddings_2 (list): Embeddings of sentences_2.

        Returns:
            tensor: Similarity score matrix computed using the embeddings.
        """
        tensor = self.model.similarity(embeddings_1, embeddings_2)
        return tensor

    @staticmethod
    def generate_score(tensor):
        """
        Static method. Computes row-wise and column-wise maximum means from the similarity score matrix.

        Args:
            tensor (torch.Tensor): Similarity score matrix.

        Returns:
            tuple: A tuple containing the mean of row-wise maximums and the mean of column-wise maximums.
        """
        max_per_row = tensor.max(dim=1).values
        r_mean = max_per_row.mean()
        max_per_col = tensor.max(dim=0).values
        c_mean = max_per_col.mean()
        return r_mean, c_mean

    @staticmethod
    def f1_score(r_mean, c_mean):
        """
        Static method. Calculates the F1 score from the row-wise and column-wise maximum means.

        Args:
            r_mean (float): Mean of row-wise maximums.
            c_mean (float): Mean of column-wise maximums.

        Returns:
            float: F1 score computed from the row-wise and column-wise means.
        """
        return (2 * r_mean * c_mean) / (r_mean + c_mean)