import nltk

nltk.download("punkt", quiet=True)


class Tokenizer:
    """
    Utility class for tokenizing text.
    """

    @staticmethod
    def tokenize(text: str):
        return nltk.word_tokenize(text)