class Validator:
    """
    Utility class for validating user input.
    """

    @staticmethod
    def validate_query(query: str) -> bool:
        """
        Validate query text.
        """

        if not query:
            return False

        if len(query.strip()) == 0:
            return False

        return True