import json
import os


class CNENTokenizer:
    def __init__(self):
        with open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "f5_tts", "g2p", "g2p", "vocab.json"),
            "r",
            encoding="utf-8",
        ) as file:
            self.phone2id: dict = json.load(file)["vocab"]
            self.phone2id = {k: int(v) + 1 for (k, v) in self.phone2id.items()}

        self.pad_token_id = 0
        self.phone2id["<PAD>"] = 0

        self.punct_token_id = len(self.phone2id)  # Punctuation marks tokens
        self.phone2id["<PUNCT>"] = len(self.phone2id)

        self.sep_token_id = len(self.phone2id)  # Sentence separation token
        self.phone2id["<SEP>"] = len(self.phone2id)

        self.id2phone = {v: k for (k, v) in self.phone2id.items()}
        from ying_singer.utils.f5_tts.g2p.g2p_generation import chn_eng_g2p

        self.tokenizer = chn_eng_g2p

    def encode(self, text):
        phone, token = self.tokenizer(text)
        token = [x + 1 for x in token]
        return token

    def decode(self, token):
        return "|".join([self.id2phone[x] for x in token])
