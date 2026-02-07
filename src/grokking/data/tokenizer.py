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
