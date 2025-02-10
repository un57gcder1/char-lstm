import torch
import torch.nn.functional as F

class RNN:
    def __init__(self, vocab_size, hidden_size, batch_size, training_data, valid_data, train = True):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        if train:
            # Data
            self.x_info = training_data[0,:,:,:]
            self.y_info = training_data[1,:,:,:]
            self.num_batches = training_data.size()[1]
            self.eval_batches = valid_data.size()[1]
            self.eval_x = valid_data[0,:,:]
            self.eval_y = valid_data[1,:,:]

            # Params
            self.wxh = torch.randn(self.vocab_size, self.hidden_size).requires_grad_()
            self.whh = torch.randn(self.hidden_size, self.hidden_size).requires_grad_()
            self.bh = torch.randn(1, self.hidden_size).requires_grad_()
            self.why = torch.randn(self.hidden_size, self.vocab_size).requires_grad_()
            self.by = torch.randn(1, self.vocab_size).requires_grad_()
        self.hidden = torch.zeros(self.batch_size, self.hidden_size, requires_grad=False)

    # Training the model
    def train(self, epochs = 30, steps_log = 10000, learning_rate = 1e-5, generate=True, decoder={}):
        print("Training for ", epochs, " epochs with a total of ", epochs*self.num_batches, " steps.")
        print("===============================================")
        for i in range(0, epochs):
            for j in range(0, self.num_batches):
                x = self.x_info[j,:,:]
                y = self.y_info[j,:,:]
                y_pred = self.step(x)
                loss = self.loss(y_pred, y)
                intLoss = loss.item()
                loss.backward()
                for p in [self.wxh, self.whh, self.bh, self.why, self.by]:
                    p.data -= p.grad*learning_rate
                    p.grad = None
                if ((j+1) % steps_log == 0):
                    print("Epoch: ", i+1, "   Step: ", j+1, "/", self.num_batches, "   Loss: ", intLoss)
                self.hidden.detach_()
            print("Epoch ", i+1, " Completed")
            with torch.no_grad():
                allLs = []
                for j in range(0, self.eval_batches):
                    x = self.eval_x[j, :]
                    y = self.eval_y[j, :]
                    y_pred = self.step(x)
                    l = self.loss(y_pred, y)
                    allLs.append(l)
                allLs = torch.tensor(allLs)
                mLoss = allLs.mean()
                mLoss = mLoss.item()
                print("Validation loss: ", mLoss)
    def loss(self, theOutput, actual):
        return -1*(actual*torch.log(theOutput)).sum()
    
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
    def generate(self, prompt, temperature = 1.0):
        with torch.no_grad():
            return self.step(prompt, temperature=temperature)
