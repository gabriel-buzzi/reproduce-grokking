import numpy as np

class Tokenizer:
    def __init__(self, max_operands_value: int):
        self.vocab = []
        self.max_operands_value = max_operands_value
        self.digits = [i for i in range(self.max_operands_value + 1)]
        self.vocab.extend(list(map(str, self.digits)))
        self.special_tokens = [
            "+",
            "=",
            "%",
            "[PAD]",
            "[START]",
            "[END]",
            "[MASK]",
        ]
        self.vocab.extend(self.special_tokens)
        self.one_hots = list(
            map(tuple, np.diag(np.full(len(self.vocab), 1)).tolist())
        )
        self.token2onehot = dict(zip(self.vocab, self.one_hots))
        self.onehot2token = dict(zip(self.one_hots, self.vocab))
        self.vocab_size = len(self.vocab)

    def _tokenize_sequence(self, text):
        if isinstance(text, str):
            tokens = text.split()

        embeddings = []
        for token in tokens:
            embeddings.append(self.token2onehot[token.strip()])

        return embeddings

    def tokenize(self, input_data):
        if isinstance(input_data, str):
            return self._tokenize_sequence(input_data)
        elif isinstance(input_data, list):
            return [self._tokenize_sequence(seq) for seq in input_data]
        else:
            raise Exception(
                "Tokenizer.tokenize() input_data is not str nor list[str]."
                f"Received {type(input_data)} instead."
            )

    def _detokenize_sequence(self, embeddings):
        tokens = []
        for embedding in embeddings:
            tokens.append(self.onehot2token[tuple(embedding)])

        # return " ".join(tokens)
        return tokens
    
    def detokenize(self, input_data):
        if isinstance(input_data[0][0], int):
            return self._detokenize_sequence(input_data)
        elif isinstance(input_data[0][0], list):
            return [self._detokenize_sequence(seq) for seq in input_data]
        else:
            raise Exception(
                "Tokenizer.tokenize() input_data is not list[int] nor list[list]."
                f"Received {type(input_data)}[{type(input_data[0])}] instead."
            )

def generate(
    mod_operand: int = 5, max_operands_value: int = 5, max_seq_length: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """ Returns a dataset with shape (n_samples, max_seq_length, vocab_size) """
    tokenizer = Tokenizer(max_operands_value)

    text_data = []
    masked_text_data = []
    attenntion_masks = []

    for a in tokenizer.digits:
        for b in tokenizer.digits:
            result = (a + b) % mod_operand

            sample = f"{a} + {b} % {mod_operand} = {result} [END] "
            masked_sample = f"[START] {a} + {b} % {mod_operand} = [MASK] [END] "

            padding_size = max_seq_length - len(masked_sample.split())
            padding = " ".join(["[PAD]"] * padding_size)
            attenntion_mask = [0]*len(masked_sample.split()) + [float("-inf")]*padding_size

            sample += padding + " [PAD]" # Additional padding token due to output shift
            masked_sample += padding

            text_data.append(sample)
            masked_text_data.append(masked_sample)
            attenntion_masks.append(attenntion_mask)

    return masked_text_data, text_data, attenntion_masks
