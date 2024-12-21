import math
import re


def entropy(string):
    """
    Calculates the Shannon entropy of a string.

    Entropy is a measure of the randomness or unpredictability of the string.
    """
    if not string:
        return 0.0  # Handle edge case for empty string

    # Get probability of each unique character in the string
    prob = [string.count(c) / len(string) for c in set(string)]

    # Calculate the Shannon entropy
    return -sum(p * math.log2(p) for p in prob)


def count_consonants(string):
    """
    Counts the number of consonants in a string.

    Consonants are defined as the letters [bcdfghjklmnpqrstvwxyz].
    """
    consonants_pattern = re.compile(r"[bcdfghjklmnpqrstvwxyz]", re.IGNORECASE)
    return len(consonants_pattern.findall(string))
