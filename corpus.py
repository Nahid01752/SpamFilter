import nltk
import ssl
from nltk.corpus import stopwords
import pathlib
from pathlib import PosixPath


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')  # download stopwords if not already downloaded

stop_words = set(stopwords.words('english'))
tokenizer = nltk.RegexpTokenizer(r'\w+')  # to remove space and punctuations


def tokenize(text):
    words = tokenizer.tokenize(text.lower())  # lowercase the text
    return [word for word in words if word not in stop_words]


def read_file(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        next(f)  # skip first line
        text = f.read()
    return tokenize(text)


def read_dataset(path):
    train = []      # list of token list and class as tuples

    # for ham
    for p in pathlib.Path(path).iterdir():  # enter to the directory
        if p == PosixPath('train/ham'):     # if ham in the directory
            for q in pathlib.Path(p).iterdir():     # get the emails
                token_n_c = (read_file(q), 'ham')   # tokenize and tag
                train.append(token_n_c)     # add to the train list as tuple
    # for ham
    for p in pathlib.Path(path).iterdir():  # enter to the directory
        if p == PosixPath('train/spam'):    # if spam in the directory
            for q in pathlib.Path(p).iterdir():  # get the emails
                token_n_c = (read_file(q), 'spam')  # tokenize and tag
                train.append(token_n_c)     # add to the train list as tuple
    return train


# train_data = read_dataset('train')
# print(train_data)
