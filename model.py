import torch
import torch.nn.functional as F
from data import TextDataLoader

# Data: Variable x Batch size x Sequence Length
# Data[0] = Batch size x Sequence Length (X), same with (Y)
class RNN:
    def __init__(self, embed_size, vocab_size, hidden_size, window, training_data, valid_data, train = True):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.window = window
        self.embed_size = embedding_size
        if train:
            # Data
            self.x_info = training_data[0]
            self.y_info = training_data[1]
            self.num_examples = training_data[0].size()[0]
            self.eval_examples = valid_data[0].size()[0]
            self.eval_x = valid_data[0]
            self.eval_y = valid_data[1]

            # Params
            self.wxh = torch.randn(self.vocab_size, self.hidden_size).requires_grad_()
            self.whh = torch.randn(self.hidden_size, self.hidden_size).requires_grad_()
            self.bh = torch.randn(1, self.hidden_size).requires_grad_()
            self.why = torch.randn(self.hidden_size, self.vocab_size).requires_grad_()
            self.by = torch.randn(1, self.vocab_size).requires_grad_()
            self.embedding = torch.randn(self.vocab_size, self.embedding_size).requires_grad_()
        self.hidden = torch.zeros(self.window, self.hidden_size, requires_grad=False)

    # Training the model
    def train(self, epochs = 30, steps_log = 10000, learning_rate = 1e-5, generate = False, 
              tdl = None, prompt = None, chars = 200, temperature = 1.0, save_best = True):
        return
    def loss(self, theOutput, actual):
        epsilon = 1e-9
        return -1*((actual*torch.log(theOutput+epsilon)).sum())
    
    # theInput: a matrix of shape batch_size x vocab_size (row, columns)
    # One RNN timestep
    def step(self, theInput, temperature = 1.0):
        self.hidden = torch.tanh(self.hidden@self.whh + theInput@self.wxh + self.bh)
        theOutput = self.hidden@self.why + self.by
        theOutput = F.softmax(theOutput/temperature, dim=-1)
        return theOutput

    # Saving the model weights, etc.
    def save(self, filename = "model.pth"):
        d = {"embedding":self.embedding,
             "wxh":self.wxh,
             "whh":self.whh,
             "bh":self.bh,
             "why":self.why,
             "by":self.by}
        torch.save(d, filename)
        return 0

    # Loading the model weights from a file
    def load(self, filename = "model.pth"):
        d = torch.load(filename)
        self.embedding = d["embedding"]
        self.wxh = d["wxh"]
        self.whh = d["whh"]
        self.bh = d["bh"]
        self.why = d["why"]
        self.by = d["by"]
        return 0
