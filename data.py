import numpy as np
import torch
from torch import tensor
import os


"""
* Takes a text file and converts it into trainable, tensor data

** textFile is the path to the train, valid files (required)
*** A datatype that provides the name (list or dict) with format
**** [train_file, valid_file], {"train": train_file, "valid": valid_file}

"""
class TextDataLoader:
    def __init__(self, textFile, split = 0.2):
        if type(textFile) == list and len(textFile) == 2:
            self.train = textFile[0]
            self.valid = textFile[1]
        elif type(textFile) == dict:
            self.train = textFile["train"]
            self.valid = textFile["valid"]
        else:
            raise Exception("""Text file provided is not a valid type.
                            Please provide a list [train_file, valid_file], 
                            or dict = {'train': train_file, 'valid': valid_file}.""")
        
        if not os.path.isfile(self.train):
            raise Exception("Training file does not have a valid path.")
        if not os.path.isfile(self.valid):
            raise Exception("Validation file does not have a valid path.")

        self.tokens = {"unk": 0}
        self.training_data = []
        self.valid_data = []

    # Reads file and tokenizes it, along with assigning a numerical value
    def tokenize(self):
        with open(self.train, 'r') as f:
            for line in f:
                for w in line:
                    if w not in self.tokens:
                        self.tokens[w] = len(self.tokens)
        return self.tokens

    def numericalize(self):
        with open(self.train, 'r') as f:
            for line in f:
                for w in line:
                    self.training_data.append(self.tokens[w])
        with open(self.valid, 'r') as f:
            for line in f:
                for w in line:
                    # Accounting for unknown tokens
                    try:
                        self.valid_data.append(self.tokens[w])
                    except:
                        self.valid_data.append(self.tokens["unk"])

        self.training_data = tensor(self.training_data)
        self.valid_data = tensor(self.valid_data)
        return (self.training_data, self.valid_data)

    def substantiate(self):
        ind_var = self.training_data[:len(self.training_data)-1]
        dep_var = self.training_data[1:]
        n = np.array([ind_var, dep_var])
        self.training_data = tensor(n)

        ind_val_var = self.valid_data[:len(self.valid_data)-1]
        dep_val_var = self.valid_data[1:]
        m = np.array([ind_val_var, dep_val_var])
        self.valid_data = tensor(m)

        return (self.training_data, self.valid_data)

    # Length tokens per batch (default is 64 tokens)
    def batch(self, length = 64):
        return
