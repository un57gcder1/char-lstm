from data import TextDataLoader
from model import RNN

EPOCHS = 50
LR = 1e-3
TEMP = 1.0
BS = 8
SL = 50
HS = 512
PROMPT = "To begin: I was quite confused and alarmed by the startling news"
CHARS = 200

t = TextDataLoader(["testing/jt.txt","testing/jv.txt"])

t.preprocess(BS, SL)

#o = t.load()

#model = RNN(t.size(), HS, WINDOW, o[0], o[1])
#model.load()
#s = model.train(learning_rate=LR, generate=True, epochs=EPOCHS, tdl=t, prompt=PROMPT, temperature=TEMP)
#model.save("50-epochs.pth")
#print(PROMPT, model.generate(prompt=PROMPT,tdl=t,temperature=TEMP))
