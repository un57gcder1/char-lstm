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
    def __init__(self, textFile, window):
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
        self.window = window 
        if self.window < 1:
            raise Exception("Window size must be greater than or equal to 1.")
        if self.window > 1000:
            raise Warning("Window size is unusually large. May cause errors.")

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

    def substantiate(self):
        ind_var = self.training_data[:len(self.training_data)-1]
        ind_var = self.__expand(ind_var, ind_var.size()[0])
        dep_var = self.training_data[self.window:]
        dep_var = dep_var.view(dep_var.size()[0], 1, dep_var.size()[1])
        self.training_data = (ind_var, dep_var)

        ind_val_var = self.valid_data[:len(self.valid_data)-1]
        ind_val_var = self.__expand(ind_val_var, ind_val_var.size()[0])
        dep_val_var = self.valid_data[self.window:]
        dep_val_var = dep_val_var.view(dep_val_var.size()[0], 1, dep_val_var.size()[1])
        self.valid_data = (ind_val_var, dep_val_var)

        return (self.training_data, self.valid_data)

    # Expanding the independent variable to window size
    # theTensor = shape [examples, vocab_size]
    def __expand(self, theTensor, examples):
        nL = []
        for i in range(examples):
            unique = theTensor[0+i:self.window+i,:]
            if (unique.size()[0] != self.window):
                break
            nL.append(unique)
        nL = torch.stack(nL)
        return nL

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
    
    def save(self, filename="db.pth"):
        d = {"training_data":self.training_data,
             "valid_data":self.valid_data,
             "tokens":self.tokens}
        torch.save(d, filename)
        return 0

    def load(self, filename="db.pth"):
        d = torch.load(filename)
        self.training_data = d["training_data"]
        self.valid_data = d["valid_data"]
        self.tokens = d["tokens"]
        return (self.training_data, self.valid_data)
