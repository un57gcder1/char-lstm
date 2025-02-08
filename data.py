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
    def __init__(self, textFile):
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

    def substantiate(self, window = 1):
        if window < 1:
            raise Exception("Window size must be greater than or equal to 1.")
        if window > 1000:
            raise Warning("Window size is unusually large. May cause errors.")
        ind_var = self.training_data[:len(self.training_data)-window]
        dep_var = self.training_data[window:]
        n = np.array([ind_var, dep_var])
        self.training_data = tensor(n)

        ind_val_var = self.valid_data[:len(self.valid_data)-window]
        dep_val_var = self.valid_data[window:]
        m = np.array([ind_val_var, dep_val_var])
        self.valid_data = tensor(m)

        return (self.training_data, self.valid_data)

    # Batch size (default is 8 tokens)
    # Note: Removes last, incomplete batch (no padding)
    def batch(self, batch_size = 8):
        if self.training_data.dim() == 2 and self.valid_data.dim() == 2:
            self.training_data = self.training_data[:,:(self.training_data.size()[1]//batch_size)*batch_size]
            self.training_data = self.training_data.view(self.training_data.size()[0], 
                               self.training_data.size()[1]//batch_size, 
                               batch_size)

            self.valid_data = self.valid_data[:,:(self.valid_data.size()[1]//batch_size)*batch_size] 
            self.valid_data = self.valid_data.view(self.valid_data.size()[0], 
                               self.valid_data.size()[1]//batch_size, 
                               batch_size)
        elif self.training_data.dim() == 1 and self.valid_Data.dim() == 1: 
            self.training_data = self.training_data[:(self.training_data.size()[0]//batch_size)*batch_size]
            self.training_data = self.training_data.view(self.training_data.size()[1]//batch_size, 
                               batch_size)

            self.valid_data = self.valid_data[:(self.valid_data.size()[0]//batch_size)*batch_size]
            self.valid_data = self.valid_data.view(self.valid_data.size()[1]//batch_size, 
                               batch_size)
        else:
            raise Exception("Training Data & Valid Data tensors are of wrong shape.")
        return (self.training_data, self.valid_data)
    def size(self):
        return len(self.tokens)
