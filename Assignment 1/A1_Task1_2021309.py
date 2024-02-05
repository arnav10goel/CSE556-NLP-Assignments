import re, string
from collections import defaultdict

class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.bpe_rules = []
        self.old_vocab = set()

    def get_stats(self):
        pairs = defaultdict(int)
        for word, freq in self.vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(self, pair):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in self.vocab:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = self.vocab[word]
        self.vocab = v_out

    def get_vocab(self, data):
        # lower case all the text
        # Remove newline characters and split into sentences
        data = [sentence.strip().lower() for sentence in data]
        vocab = defaultdict(int)
        for line in data:
            for word in line.split():
                # Remove all punctuation from the word
                word = word.translate(str.maketrans('', '', string.punctuation))
                vocab[' '.join(list(word)) + ' $'] += 1
                # Add unique characters to old_vocab
                for char in word:
                    if char not in self.old_vocab:
                        self.old_vocab.add(char)
        self.old_vocab.add('$')  # Add end of word symbol
        return vocab


    def learn_vocablury(self, data, num_merges):
        self.vocab=self.get_vocab(data)
        for _ in range(num_merges):
            pairs = self.get_stats()
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.bpe_rules.append(best)
            self.merge_vocab(best)

    def tokenize(self, sample):
        if not self.bpe_rules:
            return sample.split()  # Fallback if no vocab learned

        # Tokenize each word separately and apply merge rules
        tokens = []
        for word in sample.split():
            # Remove all punctuation from the word
            word = word.translate(str.maketrans('', '', string.punctuation))
            word_chars = ' '.join(list(word)) + ' $'  # Convert word into characters and add end-of-word symbol
            for pair in self.bpe_rules:
                bigram = re.escape(' '.join(pair))
                p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
                word_chars = p.sub(''.join(pair), word_chars)
            tokens.extend(word_chars.split())
        return tokens


    def print_merge_rules(self, filename):
        with open(filename, 'w') as file:
            for i, rule in enumerate(self.bpe_rules, 1):
                file.write(f"{rule[0]},{rule[1]}\n")

    def print_tokens(self, filename):
        with open(filename, 'w') as file:
            # Iterate over each BPE rule and add the pair to old_vocab
            for i, rule in enumerate(self.bpe_rules, 1):
                self.old_vocab.add(rule[0]+rule[1])
            sorted_tokens = sorted(self.old_vocab, key=lambda x: (len(x), x))
            # Write all tokens in sorted order to the file
            for token in sorted_tokens:
                file.write(token + '\n')

# Read corpus from file
corpus_file = "corpus.txt"
with open(corpus_file, "r") as file:
    corpus = file.readlines()

tokenizer = Tokenizer()
## To be modified according to the requirements
n = 500 # Number of merges
tokenizer.learn_vocablury(corpus, n)

# Print the merge rules to merge_rules.txt
tokenizer.print_merge_rules('merge_rules.txt')

# Generate tokens and save to tokens.txt
tokenizer.print_tokens('tokens.txt')

#Generate tokenized samples and save to tokenized_samples.txt
with open('corpus.txt', 'r') as file:
    sample=file.readlines()
    sample = [sentence.strip().lower() for sentence in sample]
    with open('tokenized_samples.txt', 'w') as file:
        for sentence in sample:
            # print(sentence)
            tokens = tokenizer.tokenize(sentence)
            file.write(','.join(tokens) + '\n')