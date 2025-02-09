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
                    la_list = [0]*self.size()
                    la_list[self.tokens[w]] = 1
                    self.training_data.append(la_list)
        with open(self.valid, 'r') as f:
            for line in f:
                for w in line:
                    la_list = [0]*self.size()
                    # Accounting for unknown tokens
                    try:
                        la_list[self.tokens[w]] = 1
                    except:
                        la_list[self.tokens["unk"]] = 1
                    self.valid_data.append(la_list)

        self.training_data = tensor(self.training_data, dtype=torch.float32)
        self.valid_data = tensor(self.valid_data, dtype=torch.float32)
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

    # Batch size (default is 1 token)
    # Note: Removes last, incomplete batch (no padding)
    def batch(self, batch_size = 1):
        if self.training_data.dim() == 3 and self.valid_data.dim() == 3:
            # Rounds down to remove last batch: ex. [2, 108, 37] --> [2, 104, 37]
            self.training_data = self.training_data[:,:(self.training_data.size()[1]//batch_size)*batch_size,:]

            # Format: 2 variables, num_batches, batch_size, vocab_size
            self.training_data = self.training_data.view(self.training_data.size()[0],
                               self.training_data.size()[1]//batch_size, batch_size,
                               self.training_data.size()[2])

            # Rounds down to remove last batch: ex. [2, 108, 37] --> [2, 104, 37]
            self.valid_data = self.valid_data[:,:(self.valid_data.size()[1]//batch_size)*batch_size, :]

            self.valid_data = self.valid_data.view(self.valid_data.size()[0],
                               self.valid_data.size()[1]//batch_size, batch_size,
                                                   self.valid_data.size()[2])
        return (self.training_data, self.valid_data)
        """
        elif self.training_data.dim() == 1 and self.valid_Data.dim() == 1: 
            self.training_data = self.training_data[:(self.training_data.size()[0]//batch_size)*batch_size]
            self.training_data = self.training_data.view(self.training_data.size()[1]//batch_size, 
                               batch_size)

            self.valid_data = self.valid_data[:(self.valid_data.size()[0]//batch_size)*batch_size]
            self.valid_data = self.valid_data.view(self.valid_data.size()[1]//batch_size, 
                               batch_size)
        """
    """else:
            raise Exception("Training Data & Valid Data tensors are of wrong shape.")"""
        #return (self.training_data, self.valid_data)

    def size(self):
        return len(self.tokens)

    def encode(self, theInput):
        output = []
        for i in theInput:
            la_list = [0]*self.size()
            # Accounting for unknown tokens
            try:
                la_list[self.tokens[i]] = 1
            except:
                la_list[self.tokens["unk"]] = 1
            output.append(la_list)
        output = tensor(output, dtype=torch.float32)
        return output

    def decode(self, output): 
        decoder = {v: k for k,v in self.tokens.items()}
        theInput = ""
        for i in output:
            value, index = torch.max(i, dim=0)
            num = index.item()
            theInput += decoder[num]
        return theInput
