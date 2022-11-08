import nltk
from nltk.corpus import stopwords
import codecs
from pathlib import PosixPath
import pathlib  # to open the directory


# define tokenize
def tokenize(text):
    # get english stopwords
    stop_words = set(stopwords.words("english"))
    tokenizer = nltk.RegexpTokenizer(r"\w+")  # to remove space and punctuations

    # tokenize words
    words = tokenizer.tokenize(text)

    # remove stopwords
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words


# define read_file
def read_file(path):
    # open the file in utf-8 format
    file = codecs.open(path, "r", encoding='utf-8', errors='ignore')

    # read the file from the second line
    next(file)
    file = file.read()

    # use tokenize function
    contents = tokenize(file)
    return contents


# print(read_file("0008.2001-02-09.kitchen.ham.txt"))


# define dataset
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


# print(read_dataset("train"))
