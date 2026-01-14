# %% [markdown]
"""
The goal is to represent a binary operation
in the form of a dataset in a way that the
operands became any discrite symbolwith no
internal structure
"""

# %%
import numpy as np

# %%
# Creating the dataset for (a + b) % p operation
# would something like:
vocab_size = 5
p = 5

a_ = [i for i in range(vocab_size)]
b_ = [i for i in range(vocab_size)]

data = [[None]*len(b_) for _ in range(len(a_))]

# each position in the matrix is a sample
# the inputs are the coordinates and 
# the output is the value of the cell
# TODO: Maybe this opeartion can be vectorized these avoid loops
for i, a in enumerate(a_):
    for j, b in enumerate(b_):
        data[i][j] = (a+b) % p

print(data)

# %%
# Below is how the data would be presented to the model
# (1 + 2) % 3 = 0, for example.

# Create the one hot encoding vectors with a
# additional size for equal signal
x1 = [0 for i in range(vocab_size + 1)]
x1[1] = 1
x2 = [0 for i in range(vocab_size + 1)]
x2[2] = 1
x3 = [0 for i in range(vocab_size + 1)]
x2[-1] = 1

y = [0 for i in range(vocab_size + 1)]
y[0] = 1

print(x1)
print(x2)
print(x3)
print()
print(y)

# %%
# Automating the dataset creating would be something
# like
p = 5
inputs = []
outputs = []
for i in range(vocab_size):
    for j in range(vocab_size):
        
        x1 = np.zeros((vocab_size + 1, ))
        x1[i] = 1

        x2 = np.zeros((vocab_size + 1, ))
        x2[j] = 1

        x3 = np.zeros((vocab_size + 1, ))
        x3[-1] = 1

        y = np.zeros((vocab_size + 1, ))
        y[(i + j) % p] = 1

        X = np.vstack([x1, x2, x3])
        # transformers models have their output
        # of the same shape as the input with tokens
        # forward shifited by 1.
        Y = np.vstack([x2, x3, y])

        inputs.append(X)
        outputs.append(Y)

inputs = np.array(inputs)
outputs = np.array(outputs)

# %%
