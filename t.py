from data import TextDataLoader
from model import RNN

BS = 64
WINDOW = 64
HS = 256
PROMPT = "To begin: I was quite confused and alarmed by the startling news"
CHARS = 1000

def join(str1, str2, limit):
    comp = str1+str2
    while len(comp) > limit:
        comp = comp[1:]
    return comp

t = TextDataLoader(["testing/jt.txt","testing/jv.txt"])

l = t.tokenize()

m = t.numericalize()

n = t.substantiate(window = WINDOW)

o = t.batch(batch_size = BS)
"""
print(m[0].shape, m[1].shape)

print(n[0].shape, n[1].shape)

print(o[0].shape, o[1].shape)

print(o[0], o[1])
"""
model = RNN(t.size(), HS, BS, o[0], o[1])
#model.load()
for i in range(30):
    model.train(epochs=1)
    print("Generating text: ")
    fill = ""
    for i in range(CHARS):
        s = t.encode(join(PROMPT, fill, limit=WINDOW))
        ns = model.generate(s)
        res = t.decode(ns)
        fill += res[len(res)-1]
    print(fill)
model.save()
