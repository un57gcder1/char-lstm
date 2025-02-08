from model import LSTM
from data import TextDataLoader

from torch import nn
from torch import optim
import numpy as np
# Global variables
TRAIN_TEXT_FILE = "testing/tt.txt"
VALID_TEXT_FILE = "testing/tv.txt"
EPOCHS = 30
BATCH_SIZE = 16

loss_fn = nn.CrossEntropyLoss(reduction="sum")

# Loading text data
data = TextDataLoader([TRAIN_TEXT_FILE, VALID_TEXT_FILE])
data.tokenize()
data.numericalize()
data.substantiate()
reals = data.batch(batch_size = BATCH_SIZE)

# Look at why n_hidden needs to be set to BATCH_SIZE
model = LSTM(data.size(), bs = BATCH_SIZE, n_hidden = BATCH_SIZE, n_layers = 2) 

optimizer = optim.Adam(model.parameters())
best_model = None
best_loss = np.inf
for epoch in range(EPOCHS):
    model.train()
    for X_batch, y_batch in reals:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    loss = 0
    with torch.no_grad():
        for X_batch, y_batch in reals:
            y_pred = model(X_batch)
            loss += loss_fn(y_pred, y_batch)
        if loss < best_loss:
            best_loss = loss
            best_model = model.state_dict()
        print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))

torch.save([best_model, char_to_int], "single-char.pth")
