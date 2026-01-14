import numpy as np

#TODO: this needs to be improved
# It might be interesting to generate data in the form of text
# and have a tokenizer that can convert from text to one-hot and
# back from one-hot to text.
def generate(
    p: int = 5, max_operands_value: int = 5, max_seq_length: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    X = []
    # vocab_size is the max possible operand value plus four,
    # for start, end, pedding, mask and equal tokens
    vocab_size = max_operands_value + 5
    start = np.zeros((vocab_size,))  # start token
    start[-1] = 1
    equal = np.zeros((vocab_size,))  # equal token
    equal[-2] = 1
    end = np.zeros((vocab_size,))  # end token
    end[-3] = 1
    pad = np.zeros((vocab_size,))  # padding token
    pad[-4] = 1
    mask = np.zeros((vocab_size, ))
    mask[-5] = 1

    for i in range(max_operands_value):
        for j in range(max_operands_value):
            x1 = np.zeros((vocab_size,))
            x2 = np.zeros((vocab_size,))
            y = np.zeros((vocab_size,))
            x1[i] = 1
            x2[j] = 1
            y[(i + j) % p] = 1

            x = np.vstack(
                [
                    start,
                    x1,
                    x2,
                    equal,
                    y,
                    end,
                    *[pad for _ in range(max_seq_length - 6)],
                ]
            )

            X.append(x)

    attention_mask = [*[0]*6, *[-float("inf")]*(max_seq_length-6)]

    X = np.array(X)

    return X, mask, attention_mask


if __name__ == "__main__":
    inputs, outputs = generate()
