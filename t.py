from data import TextDataLoader
from model import RNN

EPOCHS = 50
LR = 1e-4
TEMP = 0.5
WINDOW = 50
HS = 256
PROMPT = "Why are you so stupid? I am confused by this matter as much as y"
# "To begin: I was quite confused and alarmed by the startling news"
CHARS = 200

t = TextDataLoader(["testing/smallt.txt","testing/smallv.txt"], WINDOW)

#l = t.tokenize()

#m = t.numericalize()

#print(m[0].shape, m[1].shape)

#n = t.substantiate()

#print(n[0][0].shape, n[0][1].shape, n[1][0].shape, n[1][1].shape)

#t.save()

o = t.load()

model = RNN(t.size(), HS, WINDOW, o[0], o[1])
model.load()
s = model.train(learning_rate=LR, generate=True, epochs=EPOCHS, tdl=t, prompt=PROMPT, temperature=TEMP)
model.save("100-epochs.pth")
