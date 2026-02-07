import numpy as np

def generate_dataset(
    mod_operand: int = 5, max_operands_value: int = 5, max_seq_length: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """ Returns a dataset with shape (n_samples, max_seq_length, vocab_size) """

    text_data = []
    masked_text_data = []
    attenntion_masks = []

    target_token_idx = 6 # Which is the 6th token of sample

    for a in range(max_operands_value + 1):
        for b in range(max_operands_value + 1):
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

    return masked_text_data, text_data, attenntion_masks, target_token_idx
