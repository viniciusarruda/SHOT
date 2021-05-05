import numpy.matlib as matlib
import numpy as np
import matplotlib.pyplot as plt

TODO
TODO
TODO
TODO
TODO
TODO

def Entropy(input_):
    entropy = -input_ * np.log(input_ + 1e-5)
    entropy = np.sum(entropy, axis=0)
    return entropy 

def build(arr1, arr2, k):

    for i in range(k):
        arr1 = np.vstack((matlib.repmat(arr1, 1, arr2.shape[1]), np.repeat(arr2, arr1.shape[1])))

    valid = []
    for j in range(arr1.shape[1]):
        valid.append(np.sum(arr1[:, j]) == 1.0)

    arr1 = arr1[:, valid]

    return arr1

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(32,4))

ys = np.linspace(0.0, 1.0, num=21)

yys = build(np.array([ys]), np.array([ys]), 3)

xs = np.array(list(range(yys.shape[1])))

N = yys.shape[1]

es = Entropy(yys)

# min hline
min_es = np.min(es)
xs_min_es = xs[es == np.min(es)]
ys_min_es = xs_min_es.shape[0] * [min_es]

ax1.plot(xs_min_es, ys_min_es, 'r.')
ax1.hlines(y=np.min(es), xmin=0, xmax=N, linewidth=1, color='r')

# max hline
max_es = np.max(es)
xs_max_es = xs[es == np.max(es)]
ys_max_es = xs_max_es.shape[0] * [max_es]

ax1.plot(xs_max_es, ys_max_es, 'g.')
ax1.hlines(y=np.max(es), xmin=0, xmax=N, linewidth=1, color='g')

def append_labels(labels, a, b, c, d):
    a = int(a) if int(a) == a else a
    b = int(b) if int(b) == b else b
    c = int(c) if int(c) == c else c
    d = int(d) if int(d) == d else d
    labels.append('{},{},{},{}'.format(a, b, c, d))

labels = []
for x in xs_min_es:
    ax1.vlines(x=x, ymin=min_es, ymax=max_es, linewidth=1, color='gray')
    ax2.vlines(x=x, ymin=0, ymax=1, linewidth=1, color='gray')
    append_labels(labels, yys[0, x], yys[1, x], yys[2, x], yys[3, x])

for x in xs_max_es:
    ax1.vlines(x=x, ymin=min_es, ymax=max_es, linewidth=1, color='gray')
    ax2.vlines(x=x, ymin=0, ymax=1, linewidth=1, color='gray')
    append_labels(labels, yys[0, x], yys[1, x], yys[2, x], yys[3, x])

ax2.set_xticks(np.concatenate((xs_min_es, xs_max_es)))
ax2.set_xticklabels(labels, fontdict={'fontsize': 6})

ax1.plot(xs[:N], es)
ax2.plot(xs[:N], yys[0, :N], 'r')
ax2.plot(xs[:N], yys[1, :N], 'g')
ax2.plot(xs[:N], yys[2, :N], 'b')
ax2.plot(xs[:N], yys[3, :N], 'm')

ax1.set_ylabel('entropy')
ax2.set_ylabel('softmax')
plt.xlabel('permutation of softmax outputs')

plt.xlim(0-5, N+5)
plt.tight_layout()
plt.savefig('entropy_example.png')
