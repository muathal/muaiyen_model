from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
class ERPDataset(Dataset):
    def __init__(self, data, word2id, intent2id, slot2id):
        self.data = data
        self.word2id = word2id
        self.intent2id = intent2id
        self.slot2id = slot2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Convert strings to IDs
        # If a word isn't in vocab, give it the <UNK> ID (1)
        word_ids = [self.word2id.get(w, 1) for w in item["text"]]
        slot_ids = [self.slot2id[s] for s in item["slots"]]
        intent_id = self.intent2id[item["intent"]]

        # Return as PyTorch Tensors
        return torch.tensor(word_ids), torch.tensor(slot_ids), torch.tensor(intent_id)

def collate_fn(batch):
    # Separate the batch into text, slots, and intents
    texts = [item[0] for item in batch]
    slots = [item[1] for item in batch]
    intents = torch.tensor([item[2] for item in batch])

    # Pad the text and slots with 0s so they are all the same length
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    padded_slots = pad_sequence(slots, batch_first=True, padding_value=0)

    return padded_texts, padded_slots, intents

def build_mapping(data):
    slot_mapping = {"0":0}
    intent_mapping = {}
    for intent in data["intent"]:
        if intent not in intent_mapping:
            intent_mapping[intent] = len(intent_mapping)
    for slot in data["slot"]:
        if slot not in slot_mapping:
            slot_mapping[slot] = len(slot_mapping)
    intent_unmapping = {value:key for key, value in intent_mapping.items()}
    slot_unmapping = {value:key for key, value in slot_mapping.items()}
    return slot_mapping,intent_mapping,slot_unmapping,intent_unmapping
#building vocab throught text
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
def build_vocab(texts, min_freq=2, max_size=50000):
    counter = Counter()
    for t in texts:
        counter.update(t)

    word_mapping = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for w, c in counter.most_common():
        if c < min_freq:
            continue
        if w in word_mapping:
            continue
        word_mapping[w] = len(word_mapping)
        if len(word_mapping) >= max_size:
            break
    word_unmapping = {value:key for key, value in word_mapping.items()}
    return word_mapping,word_unmapping

