import re


class TextCleaner:
    """
    Utility class for cleaning input text.
    """

    @staticmethod
    def clean(text: str) -> str:
        """
        Clean and normalize text.
        """

        text = text.strip()

        text = re.sub(r"\s+", " ", text)

        return text