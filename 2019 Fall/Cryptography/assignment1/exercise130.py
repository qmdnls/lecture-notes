# Generate dictionary for alphabet
alph = {chr(i+97):i for i in range(0,26)}
invalph = {v: k for k, v in alph.items()}

# Permutation dictionary and its inverse
pi = {0: 23, 1: 13, 2: 24, 3: 0, 4: 7, 5: 15, 6: 14, 7: 6, 8: 25, 9: 16, 10: 22, 11: 1, 12: 19, 13: 18, 14: 5, 15: 11, 16: 17, 17: 2, 18: 21, 19: 12, 20: 20, 21: 4, 22: 10, 23: 9, 24: 3, 25: 8}
invpi = {v: k for k, v in pi.items()}

def enc(list, key):
    y = []
    for i, x in enumerate(list):
        z = (key + i - 1) % 26
        y.append(pi[x] + z % 26)
    return y

def dec(list, key):
    x = []
    for i, y in enumerate(list):
        z = (key + i - 1) % 26
        x.append(invpi[(y - z) % 26])
    return x

def string2num(c):
    num = []
    for x in list(c.lower()):
        num.append(alph[x])
    return num

def num2string(y):
    # Use the dictionary to get a string back
    p = []
    for i in y:
        p.append(invalph[i])
    p = ''.join(p).upper()
    return p

# The ciphertext
c = "WRTCNRLDSAFARWKXFTXCZRNHNYPDTZUUKMPLUSOXNEUDOKLXRMCBKGRCCURR"

# Numeric representation of string
num = string2num(c)

# Exhaustive key search over key space
for i in range(0,26):
    y = dec(num,i)
    p = num2string(y)
    print("K = " + str(i) + ": " + p)
