# NOTE: Will implement window. Batch_size = 1
# IMPORTANT: For the time being, batch_size won't be implemented!!!!!!!!!!
import torch
import torch.nn.functional as F
from data import TextDataLoader

#class RNN:
#    def __init__(self, 
#    vocab_size, # Number of unique tokens
#    hidden_size, # Number of hidden "neurons", weights in the hidden matrix (hidden_size
#                 # x hidden_size). Hidden state is batch_size x hidden_size.
#    batch_size, # Batch_size is number of items in the mini-batch
#    training_data, # Preprocessed training data using TextDataLoader
#    valid_data, # Preprocessed valid data using TextDataLoader
#    train = True)
#    
#    def train(self, epochs = 30,
#    steps_log = 10000 # logging training loss for each steps_log thing
#    learning_rate = 1e-5, # fixed learning rate for training
#    generate = False, # whether to generate text during training
#    tdl = None, # required if generate=True, a TextDataLoader
#    prompt = None, # Prompt, required if generate=True
#    window = None, # Window size, required if generate=True
#    chars = 200, # Chars to generate during training
#    temperature = 1.0, # Temperature (softmax activation)
#    save_best = True) # Save the best model during training
#
#    def loss(self,
#    theOutput, # model output
#    actual) # Actual, true output
#
#    def step(self,
#    theInput, # Processed input for one step forward
#    temperature = 1.0)
#
#    def save(self,
#    filename = 'model.pth')
#
#    def load(self,
#    filename = 'model.pth')
#
#    // encodes text, generates an output, and decodes the output
#    def generate(self, 
#    prompt, 
#    tdl, # Text data loader
#    window, 
#    chars=200, 
#    temperature = 1.0)
#
#    def __join(self,
#    str1, # String 1 to join
#    str2, # String 2 to join
#    limit) # Limit is the window size generally to keep characters of str1 and str2 the same
#    
#    def params() // Returns model parameters
#
#    def print() // Prints model parameters
#    
#    // Finding learning rate using Fast.AI technique
#    // See https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html`
#    def finder()
class RNN:
    def __init__(self, vocab_size, hidden_size, window, training_data, valid_data, train = True):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.window = window
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
        self.hidden = torch.zeros(self.window, self.hidden_size, requires_grad=False)

    # Training the model
    def train(self, epochs = 30, steps_log = 10000, learning_rate = 1e-5, generate = False, 
              tdl = None, prompt = None, chars = 200, temperature = 1.0, save_best = True):
        if save_best:
            minLoss = torch.inf
        if generate:
            assert type(tdl) == TextDataLoader
            assert type(prompt) == str
        print("Training for ", epochs, " epochs with a total of ", epochs*self.num_examples, " steps.")
        print("===============================================")
        for i in range(0, epochs):
            for j in range(0, self.num_examples):
                x = self.x_info[j,:,:]
                y = self.y_info[j,:,:]
                y_pred = self.step(x)
                y_pred = y_pred[-1,:] # Taking last output
                loss = self.loss(y_pred, y)
                intLoss = loss.item()
                loss.backward()
                for p in [self.wxh, self.whh, self.bh, self.why, self.by]:
                    p.data -= p.grad*learning_rate
                    p.grad = None
                if ((j+1) % steps_log == 0):
                    print("Epoch: ", i+1, "   Step: ", j+1, "/", self.num_examples, "   Loss: ", intLoss)
                self.hidden.detach_()
            print("Epoch ", i+1, " Completed")
            with torch.no_grad():
                allLs = []
                for j in range(0, self.eval_examples):
                    x = self.eval_x[j,:,:]
                    y = self.eval_y[j,:,:]
                    y_pred = self.step(x)
                    y_pred = y_pred[-1,:] # Taking last output: [50,64] --> [64]
                    l = self.loss(y_pred, y)
                    allLs.append(l)
                allLs = torch.tensor(allLs)
                mLoss = allLs.mean()
                mLoss = mLoss.item()
                print("Validation loss: ", mLoss)
                if generate:
                    print("Generating text: ")
                    print(self.generate(prompt=prompt, tdl=tdl, chars=chars, temperature=temperature))
            if save_best and mLoss < minLoss:
                self.save()
                print("This model saved")
                minLoss = mLoss
    def loss(self, theOutput, actual):
        epsilon = 1e-9
        return -1*((actual*torch.log(theOutput+epsilon)).sum())
    
    # theInput: a matrix of shape batch_size x vocab_size (row, columns)
    def step(self, theInput, temperature = 1.0):
        """
        Traditional feed-forward neural networK:
        res = theInput@self.wxh
        res = res@self.whh + self.bh
        res = res@self.why + self.by"""
        self.hidden = torch.tanh(self.hidden@self.whh + theInput@self.wxh + self.bh)
        theOutput = self.hidden@self.why + self.by
        theOutput = F.softmax(theOutput/temperature, dim=-1)
        return theOutput

    # Saving the model weights, etc.
    def save(self, filename = "model.pth"):
        d = {"wxh":self.wxh,
             "whh":self.whh,
             "bh":self.bh,
             "why":self.why,
             "by":self.by}
        torch.save(d, filename)
        return 0

    # Loading the model weights from a file
    def load(self, filename = "model.pth"):
        d = torch.load(filename)
        self.wxh = d["wxh"]
        self.whh = d["whh"]
        self.bh = d["bh"]
        self.why = d["why"]
        self.by = d["by"]
        return 0

    # Run inferences on the model
    def generate(self, prompt, tdl, chars=200, temperature = 1.0):
        assert type(tdl) == TextDataLoader
        fill = ""
        for i in range(chars):
            s = tdl.encode(self.__join(prompt, fill, limit=self.window))
            with torch.no_grad():
                ns = self.step(s, temperature=temperature)
            res = tdl.decode(ns)
            fill += res[len(res)-1]
        return fill
    
    def __join(self, str1, str2, limit):
        comp = str1+str2
        while len(comp) > limit:
            comp = comp[1:]
        #print(comp)
        return comp
    
    # Returns the parameters
    def params(self):
        d = {"wxh":self.wxh,
             "whh":self.whh,
             "bh":self.bh,
             "why":self.why,
             "by":self.by}
        return d

    # Prints the parameters
    def print(self):
        d = {"wxh":self.wxh,
             "whh":self.whh,
             "bh":self.bh,
             "why":self.why,
             "by":self.by}
        print(d)
        return 0

    # Finds a good learning rate
    def finder(self):
        return
