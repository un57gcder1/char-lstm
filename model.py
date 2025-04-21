import torch
import torch.nn.functional as F
from data import TextDataLoader

# Data: Variable Tuple of Batch size x Sequence Length
# Data[0] = Batch size x Sequence Length (X), same with (Y)
class RNN:
    def __init__(self, batch_size, embed_size, vocab_size, hidden_size, timesteps, training_data, valid_data, train = True):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.embed_size = embed_size
        if train:
            # Data
            self.x_info = training_data[0]
            self.y_info = training_data[1]
            self.num_examples = training_data[0].size()[0]
            self.eval_examples = valid_data[0].size()[0]
            self.eval_x = valid_data[0]
            self.eval_y = valid_data[1]

            # Params
            self.wxh = torch.randn(self.embed_size, self.hidden_size).requires_grad_()
            self.whh = torch.randn(self.hidden_size, self.hidden_size).requires_grad_()
            self.bh = torch.randn(1, self.hidden_size).requires_grad_()
            self.why = torch.randn(self.hidden_size, self.vocab_size).requires_grad_()
            self.by = torch.randn(1, self.vocab_size).requires_grad_()
            self.embedding = torch.randn(self.vocab_size, self.embed_size).requires_grad_()
        self.hidden = torch.zeros(self.batch_size, self.hidden_size, requires_grad=False)

    # Training the model
    def train(self, epochs = 40, steps_log = 1000, learning_rate = 1e-3, save_best = True, clip_value = 0.5):
        minLoss = torch.inf
        for i in range(epochs):
            print("================ EPOCH ", i+1, " =========================")
            for j in range(self.num_examples):
                x = self.x_info[j,:,:]
                y = self.y_info[j,:,:]
                y_pred = self.forward(x)
                loss = self.loss(y_pred, y)
                intLoss = loss.item()
                #print(intLoss)
                loss.backward()
                for p in [self.wxh, self.whh, self.bh, self.why, self.by, self.embedding]:
                    p.grad.data.clamp_(min=-clip_value, max=clip_value) # Gradient clipping to prevent exploding/vanishing gradient
                    p.data -= p.grad*learning_rate
                    p.grad = None
                if ((j+1) % steps_log == 0):
                    print("Epoch: ", i+1, "   Step: ", j+1, "/", self.num_examples, "   Loss: ", intLoss)
                self.hidden.detach_()
            print("Epoch ", i+1, " completed.")
            with torch.no_grad():
                allLs = []
                for j in range(self.eval_examples):
                    x = self.eval_x[j,:,:]
                    y = self.eval_y[j,:,:]
                    y_pred = self.forward(x)
                    l = self.loss(y_pred, y)
                    allLs.append(l)
                allLs = torch.tensor(allLs)
                mLoss = allLs.mean()
                mLoss = mLoss.item()
                print("Validation loss: ", mLoss)
            if save_best and mLoss < minLoss:
                self.save()
                print("This model saved")
                minLoss = mLoss

    # Actual is an input of size BS x SL
    # theOutput is expected: shape BS x SL x VS
    def loss(self, theOutput, actual):
        theOutput = self.__expand(theOutput, actual)
        epsilon = 1e-7
        return -1*((torch.log(theOutput+epsilon)).mean())
    
    def __expand(self, theOutput, actual):
        actual = actual.unsqueeze(2) # Actual --> BS x SL x 1
        theOutput = torch.gather(theOutput, dim=2, index=actual) # Getting highest prob vocab
        return theOutput

    # theInput: a matrix of size batch_size x embed_size (row, columns)
    # One RNN timestep: output batch_size x vocab_size
    def step(self, theInput, temperature = 1.0):
        self.hidden = torch.tanh(self.hidden@self.whh + theInput@self.wxh + self.bh)
        theOutput = self.hidden@self.why + self.by
        theOutput = F.softmax(theOutput/temperature, dim=-1)
        return theOutput

    # theInput: a tensor of shape batch_size x sequence_length
    # Output: batch_size x timesteps x vocab_size
    def forward(self, theInput, temperature = 1.0):
        outputs = []
        for i in range(self.timesteps):
            x = self.embedding[theInput[:, i]]
            y = self.step(x)
            outputs.append(y)
        return torch.stack(outputs, dim=1)

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

    # Right now, the input should be BS x SL
    def generate(self, theInput):
        with torch.no_grad():
            return
