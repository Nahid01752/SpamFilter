from corpus import *
from nb import SpamFilter

# Define constants
TRAIN_DIR = "train"
TEST_DIR = "test"
HAM_DIR = "ham"
SPAM_DIR = "spam"

# Instantiate SpamFilter class
sf = SpamFilter()

# Train the classifier
print(f"Training using files in {TRAIN_DIR} directory...")
train_data = read_dataset(TRAIN_DIR)
sf.train(train_data)


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
                new_line = result.replace('test//', '')
                print(new_line)
                # print(result)
            if st2 in file:
                file_name = file.replace(st2, '')  # remove part of the string to meet the requirement
                result = (file_name, email)  #
                result = ' '.join(str(value) for value in result)
                new_line = result.replace('test//', '')
                print(new_line)
                # print(result)


classify_batch(TEST_DIR, HAM_DIR, SPAM_DIR)