from data import TextDataLoader
from model import RNN

t = TextDataLoader(["testing/tt.txt","testing/tv.txt"])

l = t.tokenize()

m = t.numericalize()

n = t.substantiate(window = 5)

o = t.batch()

"""print(m[0].shape, m[1].shape)

print(n[0].shape, n[1].shape)

print(o[0].shape, o[1].shape)

print(o[0], o[1])"""

model = RNN(t.size(), 64, 1, o[0], o[1])
model.train()
model.save()
