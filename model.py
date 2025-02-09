import torch

class RNN:
    def __init__(self, vocab_size, hidden_size, batch_size, training_data, valid_data, train = True):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        if train:
            self.x_info = training_data[0,:,:]
            self.y_info = training_data[1,:,:]
            self.num_batches = training_data.size()[1]
            self.wxh = torch.randn(vocab_size, hidden_size).requires_grad_()
            self.whh = torch.randn(hidden_size, hidden_size).requires_grad_()
            self.bh = torch.randn(hidden_size, 1).requires_grad_()
            self.why = torch.randn(hidden_size, vocab_size).requires_grad_()
            self.by = torch.randn(vocab_size, 1).requires_grad_()
        self.hidden = torch.zeros(self.batch_size, self.hidden_size, requires_grad=False)

    # Training the model
    def train(self, epochs = 30, learning_rate = 1e-5):
        print("Training for ", epochs, " epochs with a total of ", epochs*num_batches, " steps.")
        for i in range(0, epochs):
            for j in range(0, num_batches):
                x = x_info[j, :]
                y = y_info[j, :]
                y_pred = self.step(x)
                loss = self.loss(y_pred, y)
                loss.backward()
                for p in [self.wxh, self.whh, self.bh, self.why, self.by]:
                    p.data -= p.grad*lr
                    p.grad.zero_()


    def loss(self, theOutput, actual):
        return ((theOutput-actual)**2).mean()
    
    # theInput: a matrix of shape 1 x vocab_size (row, columns)
    def step(self, theInput):
        """
        Traditional feed-forward neural networK:
        res = theInput@self.wxh
        res = res@self.whh + self.bh
        res = res@self.why + self.by"""
        self.hidden = torch.tanh(self.hidden@self.whh + theInput@self.wxh + self.bh)
        theOutput = self.hidden@self.why + self.by
        return theOutput

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
    def generate(self, prompt = "", temperature = 0.5):
        return
