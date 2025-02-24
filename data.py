import numpy as np
import torch
from torch import tensor
import os
import random

"""
* Takes a text file and converts it into trainable, tensor data

** textFile is the path to the train, valid files (required)
*** A datatype that provides the name (list or dict) with format
**** [train_file, valid_file], {"train": train_file, "valid": valid_file}

NOTE: Data will not be one-hot encoded due to memory constraints; change in progress.

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
        self.training_data = tensor(self.training_data, dtype=torch.float32)
        self.valid_data = tensor(self.valid_data, dtype=torch.float32)

    # Dividing into two x, y tuples for training and valid data
    def divide(self):
        x_data = self.training_data
        y_data = torch.roll(self.training_data, shifts = 1, dims = 0)
        self.training_data = (x_data, y_data)
        xval_data = self.valid_data
        yval_data = torch.roll(self.valid_data, shifts = 1, dims = 0)
        self.valid_data = (xval_data, yval_data)

    # Right now: divide() must be run before substantiate()
    def substantiate(self, batch_size, timesteps):
        factor = batch_size * timesteps
        x_data = self.training_data[0]
        y_data = self.training_data[1]

        x_data = x_data[:x_data.size()[0]-(x_data.size()[0] % factor)]
        y_data = y_data[:y_data.size()[0]-(y_data.size()[0] % factor)]

        x_data = x_data.view(x_data.size()[0]//factor, batch_size, timesteps)
        y_data = y_data.view(y_data.size()[0]//factor, batch_size, timesteps)

        xval_data = self.valid_data[0]
        yval_data = self.valid_data[1]

        xval_data = xval_data[:xval_data.size()[0]-(xval_data.size()[0] % factor)]
        yval_data = yval_data[:yval_data.size()[0]-(yval_data.size()[0] % factor)]

        xval_data = xval_data.view(xval_data.size()[0]//factor, batch_size, timesteps)
        yval_data = yval_data.view(yval_data.size()[0]//factor, batch_size, timesteps)

        self.training_data = (x_data, y_data)
        self.valid_data = (xval_data, yval_data)

    def save(self, filename="db.pth", t="tokens.pth"):
        d = {"training_data":self.training_data,
             "valid_data":self.valid_data}
        torch.save(d, filename)
        torch.save(self.tokens, t)
        return 0

    def load(self, filename="db.pth", t="tokens.pth"):
        d = torch.load(filename)
        self.training_data = d["training_data"]
        self.valid_data = d["valid_data"]
        self.tokens = torch.load(t)
        return 0

    def preprocess(self, batch_size, timesteps, save=True, **kwargs):
        self.tokenize()
        self.numericalize()
        self.divide()
        self.substantiate(batch_size, timesteps)
        if save:
            t = kwargs.get("t", "tokens.pth")
            filename = kwargs.get("filename", "db.pth")
            self.save(filename, t)
        return 0
