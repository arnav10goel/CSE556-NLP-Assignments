import numpy as np
from utils import emotion_scores
from A1_Task2_SubTask1_2_2021265 import BigramLM

# Return emotion scores for each bigram
def get_emotion_score_bigram(bigram_lm):
    """
    bigram_lm:
        instance of the BigramLM class

    This method will be used to calculate the emotion scores for each bigram in the bigram language model. The method will return two dictionaries:
    1. bigram_prob_dict: dictionary of bigram probabilities without emotion scores
    2. emotion_prob_dict: dictionary of emotion scores for each bigram

    The method will iterate through each bigram in the bigram language model and calculate the probability of the bigram without emotion scores.
    It will then calculate the emotion score for each bigram using the emotion_scores method from the utils module. The method will return the two dictionaries.
    """

    # Store the bigram probabilities without emotion scores
    bigram_prob_dict = {}
    bigram_frequencies = bigram_lm.bigram_freq
    for bigram in bigram_frequencies:
        bigram_prob_dict[bigram] = bigram_lm.get_bigram_prob(bigram[0], bigram[1])

    # Get the emotion scores for each bigram
    emotion_prob_dict = {}
    i = 0
    for bigram in bigram_frequencies:
        i += 1
        # convert bigram to a string
        if bigram[0] == '<start>':
            bigram_str = bigram[1]
        elif bigram[1] == '<end>':
            bigram_str = bigram[0]
        else:
            bigram_str = bigram[0] + ' ' + bigram[1]

        # Get the emotion score for the bigram
        emotion_prob_dict[bigram] = emotion_scores(bigram_str)
        print(i)

    return bigram_prob_dict, emotion_prob_dict


# Combine the bigram probabilities and emotion scores
def combine_pt_and_emotion(bigram_lm, bigram_prob_dict, emotion_prob_dict):
    """
    bigram_lm:
        instance of the BigramLM class
    bigram_prob_dict:
        dictionary of bigram probabilities without emotion scores
    emotion_prob_dict:
        dictionary of emotion scores for each bigram

    - This method will be used to combine the bigram probabilities and emotion scores. The method will return a dictionary of dictionaries where the keys of the
    outer dictionary are the emotions and the keys of the inner dictionaries are the previous words of the bigram.
    - The values of the inner dictionaries are dictionaries where the keys are the next words of the bigram and the values are the combined probability of the bigram
    and the emotion score.
    - The method will iterate through each bigram in the bigram language model and combine the probability of the bigram and the emotion score.
    - The method will return the dictionary of dictionaries. The outer dictionary will have the emotions as keys and the inner dictionaries as values.
    """

    bigram_frequencies = bigram_lm.bigram_freq

    # Define a dictionary for each emotion
    combined_prob_dict = {"joy": {}, "love": {}, "sadness": {}, "surprise": {}, "fear": {}, "anger": {}}

    # Initialise the dictionary with unigrams and assign an empty dictionary to each unigram
    for word in bigram_lm.total_freq:
        for emotion in combined_prob_dict:
            combined_prob_dict[emotion][word] = {}

    # Combine the probabilities and emotion scores
    for bigram in bigram_frequencies:
        # Get the previous and next word of the bigram
        prev_word = bigram[0]
        next_word = bigram[1]

        # Get the probability of the bigram
        bigram_prob = bigram_prob_dict[bigram]

        # Get the emotion scores for the bigram
        emotion_scores = emotion_prob_dict[bigram]

        for emotion in emotion_scores:
            # Get the particular emotional score for the bigram
            curr_emo = emotion['label']
            curr_emo_score = emotion['score']

            # Add the bigram probability and emotion score to the dictionary
            combined_prob_dict[curr_emo][prev_word][next_word] = (bigram_prob + curr_emo_score)

    return combined_prob_dict

# Normalise the combined probabilities
def normalise_combined_prob(combined_prob_dict):
    """
    This method will be used to normalise the combined probabilities. The method will iterate through each emotion in the combined probabilities and normalise the
    combined probabilities for each previous word. The method will return the normalised combined probabilities.
    """

    # Normalise the combined probabilities
    for emotion in combined_prob_dict:
        for prev_word in combined_prob_dict[emotion]:
            # Get the total probability for the previous word
            total_prob = sum(combined_prob_dict[emotion][prev_word].values())

            for next_word in combined_prob_dict[emotion][prev_word]:
                # Normalise the probability
                combined_prob_dict[emotion][prev_word][next_word] = combined_prob_dict[emotion][prev_word][next_word] / total_prob

    return combined_prob_dict
# Generate a sentence of a max length of 15 or end of sentence for emotion given using the bigram model, (minimum length = 7)
def generate_sentence(emotion, max_length=15, k=6, min_length=7):
    """
    emotion: (type string)
        emotion for which the sentence is to be generated
    max_length: (type int)
        maximum length of the sentence
        - Default value is 15
    k: (type int)
        top k probable next words to choose from
        - Default value is 6
    min_length: (type int)
        minimum length of the sentence
        - Default value is 7

    - This method will be used to generate a sentence for the given emotion using the bigram language model. The method will return the generated sentence.
    - The method will start with the start token and generate the next word by selecting randomly from the top k most probable words (or less if there are less than k words).
    - The method will continue to generate the next word until the end token is generated or the maximum length is reached. The method will return the generated sentence.
    - The method will also handle the minimum length of the sentence by removing the end token from the list of probable next words if the minimum length is not reached.
    - The method will also normalise the top k probabilities before choosing the next word to choose from. This method will return the generated sentence.
    """

    # Get the combined probabilities for the given emotion
    combined_prob = normalised_combined_prob[emotion]

    # Initialise the sentence with the start token
    sentence = ['<start>']

    # Get the previous word
    prev_word = sentence[-1]

    # Generate the sentence by picking the next word by selecting randomly from the top k most probable words (or less if there are less than k words)
    for i in range(max_length):

        # If start token is the previous word, choose from entire probable next words
        if prev_word == '<start>':
            next_word = np.random.choice(list(combined_prob[prev_word].keys()), p=list(combined_prob[prev_word].values()))

        # Else, choose from top k probable next words
        else:
            # Get the list of all probable next words
            next_words = list(combined_prob[prev_word].keys())

            # Get the list of all probable next words' probabilities
            next_words_prob = list(combined_prob[prev_word].values())

            # If minimum length is not reached, remove '<end>' from the list of probable next words (if it exists)
            if (len(sentence)-1 < min_length) and ('<end>' in next_words):
                end_index = next_words.index('<end>')
                next_words.pop(end_index)
                next_words_prob.pop(end_index)

            # If there are no probable next words, break
            if len(next_words) == 0:
                break

            # Get the top min(k, len(next_words)) probable next words
            if len(next_words) < k:
                top_k_words = np.array(next_words)
                top_k_prob = np.array(next_words_prob)

            else:
                top_k_words = np.array(next_words)[np.argsort(next_words_prob)[-k:]]
                top_k_prob = np.array(next_words_prob)[np.argsort(next_words_prob)[-k:]]

            # Normalise the top k probabilities
            top_k_prob = top_k_prob / sum(top_k_prob)

            # Choose the next word
            next_word = np.random.choice(top_k_words, p=top_k_prob)

        # If the next word is the end token, break
        if next_word == '<end>':
            break
        # else, append the next word to the sentence
        sentence.append(next_word)

        # Update the previous word
        prev_word = next_word

    # Return the sentence
    return ' '.join(sentence[1:])


# Load the bigram language model from a pickle file
bigram_lm2 = BigramLM.load_model('bigram_lm.pkl')

# Print top 5 most probable bigrams without smoothing
print('Top 5 most probable bigrams without smoothing: ')
bigram_prob = {}
for bigram in bigram_lm2.bigram_freq:
    bigram_prob[bigram] = bigram_lm2.get_bigram_prob(bigram[0], bigram[1])

for bigram in sorted(bigram_prob, key=bigram_prob.get, reverse=True)[:5]:
    print(f"{bigram} :  {bigram_prob[bigram]}")

print("-------------------")
print()
# Print top 5 most probable bigrams with laplace smoothing
print('Top 5 most probable bigrams with laplace smoothing: ')

# First find the probability of each bigram with laplace smoothing
bigram_prob = {}
for bigram in bigram_lm2.bigram_freq:
    bigram_prob[bigram] = bigram_lm2.get_bigram_prob_laplace(bigram[0], bigram[1])

# Then sort the bigrams by probability and print the top 5
for bigram in sorted(bigram_prob, key=bigram_prob.get, reverse=True)[:5]:
    print(f"{bigram} :  {bigram_prob[bigram]}")

print("-------------------")
print()

# Print top 5 most probable bigrams with kneser-ney smoothing
print('Top 5 most probable bigrams with kneser-ney smoothing: ')
bigram_prob_kn = {}
for bigram in bigram_lm2.bigram_freq:
    bigram_prob_kn[bigram] = bigram_lm2.kneser_ney_smoothing(bigram[0], bigram[1])

for bigram in sorted(bigram_prob_kn, key=bigram_prob_kn.get, reverse=True)[:5]:
    print(f"{bigram} :  {bigram_prob_kn[bigram]}")

# Print the size of the bigram frequency dictionary
print(len(bigram_lm2.bigram_freq))

# Get the emotion scores for each bigram
bigram_prob_dict, emotion_prob_dict = get_emotion_score_bigram(bigram_lm2)

# Combine the probabilities and emotion scores
combined_prob_dict = combine_pt_and_emotion(bigram_lm2, bigram_prob_dict, emotion_prob_dict)

# Call the normalise_combined_prob function
normalised_combined_prob = normalise_combined_prob(combined_prob_dict)
