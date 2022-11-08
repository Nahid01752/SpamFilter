""" to train a naive bayes spam filter from a dataset of emails tokenize(), read_file(),
read_dataset() from corpus.py module and class SpamFilter() from nb.py module are imported.
corpus.py, nb.py, main.py, training dataset, testing dataset have
to be in the same directory. Inside both train and test set, there should be ham and spam.
Also to classify a single email, the email should be in the same directory."""

import pathlib  # to open the directory

# from corpus import tokenize, read_file, read_dataset
from corpus import tokenize, read_file, read_dataset

# from nb import SpamFilter
from nb import SpamFilter

# create an instance of the class SpamFilter
sf = SpamFilter()

# call train method on sf and train the dataset
print("please change the training directory name to 'train' if it's not!")
print("training...")
sf.train(read_dataset("train"))

print("please replace the sample file to classify a single email!")
print("classifying an email into spam or ham...")
#  User needs to replace the sample file with another file
print(sf.classify("0020.1999-12-14.kaminski.ham.txt"))  # call classify method and classify a single email.


# batch testing
def classify_batch(path1, st1, st2):
    for path2 in pathlib.Path(path1).iterdir():  # enter to the path1
        for file in pathlib.Path(path2).iterdir():   # enter to the path2
            email = sf.classify(file)     # classify the file
            email = ' '.join(str(value) for value in email)  # convert tuple to the string
            file = str(file)      # convert file_name to the string
            if st1 in file:
                file_name = file.replace(st1, '')   # remove part of the string to meet the requirement
                result = (file_name, email)  #
                result = ' '.join(str(value) for value in result)
                print(result)
            if st2 in file:
                file_name = file.replace(st2, '')  # remove part of the string to meet the requirement
                result = (file_name, email)  #
                result = ' '.join(str(value) for value in result)
                print(result)


# call classify_batch function
print("please change the testing directory name to 'test' if it's not!")
print("testing batch...")
classify_batch("test", 'test/ham/', 'test/spam/')
