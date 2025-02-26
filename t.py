from data import TextDataLoader
from model import RNN

EPOCHS = 50
LR = 1e-3
TEMP = 1.0
ES = 128
BS = 8
SL = 50
HS = 512
PROMPT = "To begin: I was quite confused and alarmed by the startling news"
CHARS = 200

t = TextDataLoader(["testing/jt.txt","testing/jv.txt"])

#t.preprocess(BS, SL)

t.load()

print(t.training_data[0].shape, t.training_data[1].shape, t.valid_data[0].shape, t.valid_data[1].shape)
model = RNN(BS, ES, len(t.tokens), HS, SL, t.training_data, t.valid_data)
#model.load()
s = model.train()
model.save("50-epochs.pth")
#print(PROMPT, model.generate(prompt=PROMPT,tdl=t,temperature=TEMP))
