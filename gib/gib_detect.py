#!/usr/bin/python

import pickle
import gib_detect_train


def main():
    # Load the pre-trained model
    try:
        with open('gib_model.pki', 'rb') as model_file:
            model_data = pickle.load(model_file)
    except FileNotFoundError:
        print("Error: Model file 'gib_model.pki' not found. Please train the model first.")
        return

    model_mat = model_data['mat']
    threshold = model_data['thresh']

    print("Gibberish Detection Model Loaded.")
    print("Type a string to check if it's gibberish. Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter text: ")

        # Exit condition
        if user_input.lower() == "exit":
            print("Exiting.")
            break

        # Check if the input text is gibberish
        is_gibberish = gib_detect_train.avg_transition_prob(user_input, model_mat) <= threshold
        print(f"Gibberish: {is_gibberish}\n")


if __name__ == "__main__":
    main()
