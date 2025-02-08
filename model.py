# Module containing the files for the LSTM model
import torch

class LSTM:
    def __init__(self, vocab_size, hidden_size, training_data, valid_data, train = True):
        if train:
            self.x = training_data[0,:,:]
            self.y = training_data[1,:,:]
            self.wxh = torch.randn(vocab_size, hidden_size).requires_grad_()
            self.whh = torch.randn(hidden_size, hidden_size).requires_grad_()
            self.bh = torch.randn(hidden_size, 1).requires_grad_()
            self.why = torch.randn(hidden_size, vocab_size).requires_grad_()
            self.by = torch.randn(vocab_size, 1).requires_grad_()
    
    # Saving the model weights, etc.
    def save(self, filename = "model.pth"):
        return

    # Loading the model weights from a file
    def load(self, filename = "model.pth"):
        return

    # Evaluating on validation set during training
    def evaluate(self):
        return

    # Run inferences on the model
    def infer(self, prompt = "", temperature = 0.5):
        return

    # Training the model
    def train(self, epochs = 30, learning_rate = 1e-5):
        return
