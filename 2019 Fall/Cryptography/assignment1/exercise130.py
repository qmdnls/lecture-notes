# The ciphertext
c = "WRTCNRLDSAFARWKXFTXCZRNHNYPDTZUUKMPLUSOXNEUDOKLXRMCBKGRCCURR"

# Generate dictionary for alphabet
alph = {chr(i+97):i for i in range(0,26)}
invalph = {v: k for k, v in alph.items()}

# Use the dictionary to convert the string to a list of numbers we can permute
cnum = []
for x in list(c.lower()):
    cnum.append(alph[x])

# Permutation dictionary and its inverse
pi = {0: 23, 1: 13, 2: 24, 3: 0, 4: 7, 5: 15, 6: 14, 7: 6, 8: 25, 9: 16, 10: 22, 11: 1, 12: 19, 13: 18, 14: 5, 15: 11, 16: 17, 17: 2, 18: 21, 19: 12, 20: 20, 21: 4, 22: 10, 23: 9, 24: 3, 25: 8}
invpi = {v: k for k, v in pi.items()}

def enc(list, key):
    y = []
    for i, x in enumerate(list):
        z = (key + i - 1) % 26
        y.append(perm(x) + z % 26)
    return y

def dec(list, key):
    x = []
    for i, y in enumerate(list):
        z = (key + i - 1) % 26
        x.append(invperm((y - z) % 26))
    return x

def perm(x):
    return pi[x]

def invperm(x):
    return invpi[x]

def stringtonum(c):
    num = []
    for x in list(c.lower()):
        num.append(alph[x])
    return num

def numtostring(y):
    # Use the dictionary to get a string back
    p = []
    for i in y:
        p.append(invalph[i])
    p = ''.join(p).upper()
    return p

a = "teststring"
print(a)
anum = stringtonum(a)
print(anum)
y = enc(anum,17)
print(y)
x = dec(y,17)
print(x)
p =numtostring(x)
print(p)

# Exhaustive key search over key space
for i in range(0,26):
    y = dec(cnum,i)
    p = numtostring(y)
    print(str(i) + ". " + p)
