from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Download requirements once
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    # We return a LIST to keep slot alignment perfect
    return [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]
def preprocess_json_tokens(text_list, slot_list):

    #lemmatization on the whole list
    # (Doing it before punctuation removal preserves context for POS tagging)
    lemmatized_text = lemmatize_tokens(text_list)

    clean_text = []
    clean_slots = []

    #filter out punctuation and lowercase
    for word, slot in zip(lemmatized_text, slot_list):
        word = word.lower()

        if word in ['.', ',', '?', '!', ';', ':']:
            continue

        clean_text.append(word)
        clean_slots.append(slot)

    return clean_text, clean_slots
def clean_entire_dataset(raw_data):
    #Safely handle DataFrames, or raw Lists
    if isinstance(raw_data, pd.DataFrame):
        # If it's a Pandas DataFrame, convert it to a list of dicts!
        data_list = raw_data.to_dict(orient='records')
    else:
        # It's already a list of dictionaries
        data_list = raw_data

    cleaned_dataset = []
    dropped = 0

    #Now loop over the safely loaded list of dictionaries
    for item in data_list:
        clean_text, clean_slots = preprocess_json_tokens(item["text"], item["slots"])

        # Enforce the strict length check immediately
        if len(clean_text) == len(clean_slots):
            cleaned_dataset.append({
                "text": clean_text,
                "slots": clean_slots,
                "intent": item["intent"]
            })
        else:
            dropped += 1
    return pd.DataFrame(cleaned_dataset)
#preprocess_inference_text for deployment only
def preprocess_inference_text(raw_string):
    #Simple tokenization
    text_list = raw_string.split()

    #Lemmatize
    lemmatized_text = lemmatize_tokens(text_list)

    clean_text = []

    #Punctuation removal & lowercasing
    for word in lemmatized_text:
        word = word.lower()
        # Clean punctuation
        word = word.replace('.', '').replace(',', '').replace('?', '').replace('!', '')

        if word == '':
            continue

        clean_text.append(word)

    return clean_text

class ERPDataset(Dataset):
    def __init__(self, data, word2id, slot2id,intent2id,):
        if isinstance(data, pd.DataFrame):
            self.data = data.to_dict(orient='records')
        else:
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
    for slots in data["slots"]:
        for slot in slots:
            if slot not in slot_mapping:
                slot_mapping[slot] = len(slot_mapping)
    intent_unmapping = {value:key for key, value in intent_mapping.items()}
    slot_unmapping = {value:key for key, value in slot_mapping.items()}
    return slot_mapping,intent_mapping,slot_unmapping,intent_unmapping
#building vocab through text
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
