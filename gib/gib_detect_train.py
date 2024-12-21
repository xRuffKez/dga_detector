#!/usr/bin/python

import math
import pickle

# Define the accepted characters for the model
ACCEPTED_CHARS = 'abcdefghijklmnopqrstuvwxyz '

# Create a mapping of characters to their indices
POS = {char: idx for idx, char in enumerate(ACCEPTED_CHARS)}


def normalize(line):
    """
    Normalize the input by filtering only accepted characters.
    This reduces the model size by ignoring punctuation and infrequent symbols.
    """
    return [c.lower() for c in line if c.lower() in ACCEPTED_CHARS]


def ngram(n, text):
    """
    Generate n-grams from the input text after normalizing.
    
    Args:
        n (int): Length of the n-gram.
        text (str): Input text.
        
    Yields:
        str: n-grams of the input text.
    """
    filtered = normalize(text)
    for start in range(len(filtered) - n + 1):
        yield ''.join(filtered[start:start + n])


def avg_transition_prob(text, log_prob_mat):
    """
    Calculate the average transition probability of the text using the bigram model.

    Args:
        text (str): Input text to analyze.
        log_prob_mat (list): Log probability matrix for bigram transitions.

    Returns:
        float: Average transition probability.
    """
    log_prob = 0.0
    transition_ct = 0

    for a, b in ngram(2, text):
        log_prob += log_prob_mat[POS[a]][POS[b]]
        transition_ct += 1

    # Convert log probabilities back to probabilities
    return math.exp(log_prob / (transition_ct or 1))


def train():
    """
    Train the bigram model and save it as a pickle file.
    """
    k = len(ACCEPTED_CHARS)

    # Initialize bigram counts with smoothing (prior of 10 for each pair)
    counts = [[10 for _ in range(k)] for _ in range(k)]

    # Count bigram transitions from a large text corpus
    with open('big.txt', 'r') as f:
        for line in f:
            for a, b in ngram(2, line):
                counts[POS[a]][POS[b]] += 1

    # Normalize the counts to log probabilities
    for i, row in enumerate(counts):
        row_sum = float(sum(row))
        for j in range(len(row)):
            row[j] = math.log(row[j] / row_sum)

    # Evaluate good and bad phrases
    good_probs = []
    with open('good.txt', 'r') as f:
        good_probs = [avg_transition_prob(line, counts) for line in f]

    bad_probs = []
    with open('bad.txt', 'r') as f:
        bad_probs = [avg_transition_prob(line, counts) for line in f]

    # Ensure the model can distinguish good from bad phrases
    assert min(good_probs) > max(bad_probs), "Model failed to distinguish good and bad phrases."

    # Calculate the threshold between good and bad inputs
    thresh = (min(good_probs) + max(bad_probs)) / 2

    # Save the model
    model = {'mat': counts, 'thresh': thresh}
    with open('gib_model.pki', 'wb') as f:
        pickle.dump(model, f)

    print("Model training complete. Saved as 'gib_model.pki'.")


if __name__ == '__main__':
    train()
