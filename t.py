from data import TextDataLoader

t = TextDataLoader(["testing/tt.txt","testing/tv.txt"])

l = t.tokenize()

m = t.numericalize()

n = t.substantiate()

o = t.batch()

print(m[0].shape, m[1].shape)

print(n[0].shape, n[1].shape)

print(o[0].shape, o[1].shape)

print(o[0], o[1])
