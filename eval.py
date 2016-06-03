r1 = open('a2a.t').readlines()
r2 = open('a2a.s').readlines()
tp, tn, fp, fn = 0, 0, 0, 0
for i in xrange(len(r1)):
    y_star = float(r2[i].split()[0])
    y = float(r1[i].split()[0])
    if y == 1 and y_star == 1:
        tp += 1
    elif y == 1 and y_star == -1:
        fn += 1
    elif y == -1 and y_star == 1:
        fp += 1
    elif y == -1 and y_star == -1:
        tn += 1
    else:
        raise ValueError
p = 1.0 * tp / (tp + fp + 0.0000001)
r = 1.0 * tp / (tp + fn + 0.0000001)
f = 2.0 * p * r / (p + r + 0.0000001)
print p
print r
print f

