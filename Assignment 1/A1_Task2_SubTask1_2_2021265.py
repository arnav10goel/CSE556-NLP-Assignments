import re
import pickle

# Implement the Bigram Language Model Class
class BigramLM:
    def __init__(self):
        """
        bigram_freq:
            dictionary of bigram frequencies
        total_freq:
            dictionary of unigram frequencies
        vocab:
            set of all unique words in the corpus
        vocab_size:
            size of the vocabulary
        corpus_size:
            size of the corpus
        corpus:
            list of sentences in the corpus
        labels:
            list of labels for each sentence in the corpus

        This class will be used to train a bigram language model on a given corpus and labels. The bigram language model will be used to
        calculate the probability of a bigram with and without smoothing. This is a simple language model that uses the conditional probability of a
        word given the previous word to predict the next word in a sentence. This is the init method.
        """

        self.bigram_freq = {}
        self.total_freq = {}
        self.vocab = set()
        self.vocab_size = 0
        self.corpus_size = 0
        self.corpus = []
        self.labels = []

    # Preprocess the text
    def preprocess_text(self, text):
        """
        text: input text to be preprocessed

        This method will be used to preprocess the input text. The preprocessing steps include:
        1. Converting the text to lowercase
        2. Removing all non-alphanumeric characters
        """

        # Convert to lowercase
        text = text.lower()

        # Remove all non-alphanumeric characters
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text

    # Train the bigram language model
    def train_lm(self, corpus, labels):
        """
        corpus:
            .txt file containing the corpus
        labels:
            .txt file containing the labels for each sentence in the corpus

        This method will be used to train the bigram language model on the given corpus and labels. The method will read the corpus and labels
        from the given files and calculate the bigram and unigram frequencies. It will also calculate the vocabulary size and the corpus size.
        While training the language model, the method will also handle the start and end of sentences by including the special tokens <start> and <end>.
        """

        # Open the corpus and labels file and read it
        with open(corpus, 'r', encoding='utf-8') as corpus_file, open(labels, 'r', encoding='utf-8') as labels_file:
            for line in corpus_file:
                tokens = self.preprocess_text(line).split()
                self.corpus.append(tokens)

            for line in labels_file:
                self.labels.append(line.strip())

        # Get the vocabulary
        self.vocab = set([word for sentence in self.corpus for word in sentence])

        # Get the vocabulary size
        self.vocab_size = len(self.vocab)

        # Get the corpus size
        self.corpus_size = len(self.corpus)

        # Get the bigram and unigram frequencies
        for sentence in self.corpus:
            for i in range(len(sentence) - 1):
                # Get the bigram frequencies
                bigram = (sentence[i], sentence[i + 1])
                self.bigram_freq[bigram] = self.bigram_freq.get(bigram, 0) + 1

                # Get the unigram frequencies
                unigram = sentence[i]
                self.total_freq[unigram] = self.total_freq.get(unigram, 0) + 1

                # Handling end of sentence
                if(i == len(sentence) - 2):
                    unigram = sentence[i + 1]
                    self.total_freq[unigram] = self.total_freq.get(unigram, 0) + 1

                    # end of sentence bigram
                    bigram = (sentence[i + 1], '<end>')
                    self.bigram_freq[bigram] = self.bigram_freq.get(bigram, 0) + 1

                # Handling start of sentence
                if(i == 0):
                    bigram = ('<start>', sentence[i])
                    self.bigram_freq[bigram] = self.bigram_freq.get(bigram, 0) + 1
                    unigram = '<start>'
                    self.total_freq[unigram] = self.total_freq.get(unigram, 0) + 1

    @staticmethod
    # Save the bigram language model to a pickle file
    def save_model(model, filename):
        with open(filename, 'wb') as file:
            pickle.dump(model, file)


    @staticmethod
    # Load the bigram language model from a pickle file
    def load_model(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    # Get the probability of a bigram without smoothing
    def get_bigram_prob(self, previous_word, next_word):
        """
        previous_word: (type string)
            previous word in the bigram
        next_word: (type string)
            next word in the bigram

        This method will be used to calculate the probability of a bigram without smoothing. The probability of a bigram is calculated as the
        frequency of the bigram divided by the frequency of the previous word. If the bigram is not present in the corpus, the method will return 0.
        """

        bigram = (previous_word, next_word)
        # If the bigram is not present in the corpus, return 0
        if bigram not in self.bigram_freq:
            return 0
        else:
            return self.bigram_freq[bigram] / self.total_freq[previous_word]

    """
    Q2 - Laplace Smoothing
    """
    # Get the probability of a bigram with laplace smoothing
    def get_bigram_prob_laplace(self, previous_word, next_word):
        """
        previous_word: (type string)
            previous word in the bigram
        next_word: (type string)
            next word in the bigram

        This method will be used to calculate the probability of a bigram with laplace smoothing. The probability of a bigram is calculated as the
        frequency of the bigram plus 1 divided by the frequency of the previous word plus the vocabulary size. If the bigram is not present in the corpus,
        the method will return 1 divided by the frequency of the previous word plus the vocabulary size.
        """

        bigram = (previous_word, next_word)

        # Retrieve frequency of bigram and add 1
        bigram_freq_smoothed = self.bigram_freq.get(bigram, 0) + 1

        # Retrieve frequency of previous word and add the vocabulary size
        previous_word_freq_smoothed = self.total_freq.get(previous_word, 0) + self.vocab_size

        # Return the probability
        return bigram_freq_smoothed / previous_word_freq_smoothed


    """
    Q2 - Kneser-Ney Smoothing
    """
    # Get the probability of a bigram with kneser-ney smoothing
    def kneser_ney_smoothing(self, previous_word, next_word, d=0.5):
        """
        previous_word: (type string)
            previous word in the bigram
        next_word: (type string)
            next word in the bigram
        discount: (type float)
            discount value for the kneser-ney smoothing
            - Default value is 0.75

        - This method will be used to calculate the probability of a bigram with kneser-ney smoothing.
        - The probability of a bigram is calculated as the discounted probability of the bigram plus the continuation probability of the next word
        given the previous word.
        - The discounted probability of the bigram is calculated as the maximum of the bigram frequency minus the discount value divided by the frequency
        of the previous word.
        - The continuation probability of the next word given the previous word is calculated as the maximum of the continuation count minus the discount value
        divided by the unique continuations.
        - The unique continuations is the number of unique words that follow the previous word. The continuation count is the number of times the next word follows
        the previous word. The continuation probability is multiplied by the lambda context, which is the discount value times the continuation count divided by the
        frequency of the previous word. The method will return the probability of the bigram.
        """

        # Get the bigram
        bigram = (previous_word, next_word)

        # Get the continuation count
        continuation_count = sum(1 for (_, w2) in self.bigram_freq.items() if w2 == next_word)

        # Get the unique continuations
        unique_continuations = len(set(w2 for (_, w2) in self.bigram_freq))

        # Get the continuation probability
        continuation_probability = max((continuation_count - d) / unique_continuations, 0)

        # Get the count of the bigram and the previous word
        count_bigram = self.bigram_freq.get(bigram, 0)
        count_unigram = self.total_freq.get(previous_word, 0)

        # Get the lambda context and the discounted probability of the bigram and return the final probability
        continuation_count_ = sum(1 for (w2, _) in self.bigram_freq.items() if w2 == previous_word)
        lambda_context = d * continuation_count_ / count_unigram if count_unigram > 0 else 0
        discounted = max(count_bigram - d, 0) / count_unigram if count_unigram > 0 else 0

        # Final probability
        prob = discounted + lambda_context * continuation_probability
        return prob
    