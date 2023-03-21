from corpus import read_file


class SpamFilter:
    def __init__(self):
        # store weighted tokens
        self.weighted_ham = {}
        self.weighted_spam = {}

        # define train for training corpus data
    def train(self, emails):
        # create dictionary of tokens with frequency for both 'spam' and 'ham'
        unique_ham = {}
        unique_spam = {}
        for tup_x, tup_y in emails:     # emails is a tuple of list of tokens and classification tag
            if tup_y == 'ham':
                for ch in tup_x:
                    unique_ham.setdefault(ch, 0)  # start value counting from 0
                    unique_ham[ch] = unique_ham[ch] + 1
            if tup_y == 'spam':
                for ch in tup_x:
                    unique_spam.setdefault(ch, 0)
                    unique_spam[ch] = unique_spam[ch] + 1

        # computing ham tokens and storing their weights
        for ham_k, ham_v in unique_ham.items():
            if ham_k in unique_spam.keys():     # if token in spam
                spam_v = unique_spam[ham_k]     # get the weight from spam for the same key
                ham_p = (ham_v + 1) / (ham_v + spam_v)  # add 1 smoothing
                self.weighted_ham[ham_k] = ham_p    # dictionary called from the constructor
            else:
                ham_p = (ham_v + 1) / ham_v     # if token is not in spam
                self.weighted_ham[ham_k] = ham_p

        # computing spam tokens and storing their weights
        for spam_k, spam_v in unique_spam.items():
            if spam_k in unique_ham.keys():     # if token in ham
                ham_v = unique_ham[spam_k]      # get the weight from ham for the same key
                spam_p = (spam_v + 1) / (spam_v + ham_v)    # add 1 smoothing
                self.weighted_spam[spam_k] = spam_p     # dictionary called from the constructor
            else:
                spam_p = (spam_v + 1) / spam_v        # if token is not in ham
                self. weighted_spam[spam_k] = spam_p
        # return dict(sorted(self.weighted_spam.items(), key = itemgetter(1), reverse=True)[:10])

    # classify a given email to its appropriate class
    def classify(self, email):
        # collect tokens from test file
        test_tokens = read_file(email)

        # test dictionary to store tokens and their frequency
        test_dict = {}
        for token in test_tokens:
            test_dict.setdefault(token, 0)
            test_dict[token] = test_dict[token] + 1

        # probability in ham
        ham_weight = 1
        sub_ham_weight = 1
        p_in_ham = 0.5      # prior probability
        for test_key, test_value in test_dict.items():
            if test_key in self.weighted_ham:
                test_ham = self.weighted_ham[test_key]  # get the value of the test token
                ham_weight *= test_ham      # using p1*p2*p3... equation
                sub_ham_weight *= (1 - test_ham)
                p_in_ham *= (ham_weight / (ham_weight + sub_ham_weight))
            else:
                p_in_ham *= (test_value / test_value)

        # probability in spam
        spam_weight = 1
        sub_spam_weight = 1
        p_in_spam = 0.5
        for test_key, test_value in test_dict.items():
            if test_key in self.weighted_spam:    # if test_token in weighted spam dictionary
                test_spam = self.weighted_spam[test_key]
                spam_weight *= test_spam
                sub_spam_weight *= (1 - test_spam)  # using (1-p1), (1-p2)....
                p_in_spam *= (spam_weight / (spam_weight + sub_spam_weight))
            else:
                p_in_spam *= (test_value / test_value)

        # classification
        if p_in_ham > p_in_spam:
            ham = (p_in_ham, 'ham')     # score and classification tag as tuple
            return ham
        else:
            spam = (p_in_spam, 'spam')
            return spam
