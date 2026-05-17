import torch
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from Model.Model import SGM
import json
from pathlib import Path
# Ensure NLTK resources are available on the deployment server
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)


class InferenceEngine:
    def __init__(self, model_path, model_class, word2id, intent_unmapping, slot_unmapping, device='cpu'):
        self.device = torch.device(device)
        self.word2id = word2id
        self.intent_unmapping = intent_unmapping
        self.slot_unmapping = slot_unmapping
        self.lemmatizer = WordNetLemmatizer()

        # Instantiate the model architecture and load the trained weights
        self.model = model_class
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # eval mode because we don't want to calculate gradient and waste resources

        print("Inference Engine Loaded and Ready!")

    def _get_wordnet_pos(self, word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    def _lemmatize_tokens(self,tokens):
        lemmatizer = WordNetLemmatizer()
        # We return a LIST to keep slot alignment perfect
        return [lemmatizer.lemmatize(w, self._get_wordnet_pos(w)) for w in tokens]
    def _preprocess_text(self, raw_string):
        # Simple tokenization
        text_list = raw_string.split()

        # Lemmatize
        lemmatized_text = self._lemmatize_tokens(text_list)

        clean_text = []

        # Punctuation removal & lowercasing
        for word in lemmatized_text:
            word = word.lower()
            # Clean punctuation
            word = word.replace('.', '').replace(',', '').replace('?', '').replace('!', '')

            if word == '':
                continue

            clean_text.append(word)

        return clean_text

    def _extract_entities(self, tokens, predicted_slots):
        entities = {}
        current_entity_type = None
        current_entity_words = []

        for word, slot in zip(tokens, predicted_slots):
            if slot == 'O':
                # If we just finished reading an entity, save it
                if current_entity_type:

                    entities[current_entity_type] = " ".join(current_entity_words)
                    current_entity_type = None
                    current_entity_words = []
                continue

            # If it's a B- or I- tag
            prefix, entity_type = slot.split('-')

            if prefix == 'B':
                # Save previous entity if one exists
                if current_entity_type:
                    entities[current_entity_type] = " ".join(current_entity_words)
                current_entity_type = entity_type
                current_entity_words = [word]
            elif prefix == 'I' and current_entity_type == entity_type:
                current_entity_words.append(word)

        # Catch the last entity if the sentence ends on a slot
        if current_entity_type:
            entities[current_entity_type] = " ".join(current_entity_words)

        return entities

    def predict(self, raw_text):
        """
        The main public function the backend team will call.
        """
        # 1. Clean the text
        tokens = self._preprocess_text(raw_text)
        if not tokens:
            return {"intent": "unknown", "entities": {}}

        # 2. Convert to IDs using <UNK> (1) for unknown words
        word_ids = [self.word2id.get(w, 1) for w in tokens]

        # 3. Convert to Tensor and add batch dimension (shape: [1, seq_length])
        input_tensor = torch.tensor([word_ids], dtype=torch.long).to(self.device)

        # 4. Forward Pass
        with torch.no_grad():
            slot_logits,intent_logits = self.model(input_tensor)

        # 5. Get highest probability predictions
        predicted_intent_id = torch.argmax(intent_logits, dim=1).item()
        predicted_slot_ids = torch.argmax(slot_logits, dim=2)[0].tolist()

        # 6. Unmap IDs back to human-readable strings
        predicted_intent = self.intent_unmapping[predicted_intent_id]
        predicted_slots = [self.slot_unmapping[s_id] for s_id in predicted_slot_ids]

        # 7. Extract the clean entities
        extracted_entities = self._extract_entities(tokens, predicted_slots)

        # Return a dictionary
        return {
            "intent": predicted_intent,
            "entities": extracted_entities,
            "tokens_analyzed": tokens  # Helpful for backend debugging
        }
#SERVER STARTUP
def load_deployment_vocabularies(load_path=".."):
    # the default path should be changed to fit the deployment location
    with open(f"{load_path}/word2id.json", "r", encoding="utf-8") as f:
        word_mapping = json.load(f)

    with open(f"{load_path}/intent2id.json", "r", encoding="utf-8") as f:
        intent_mapping = json.load(f)

    with open(f"{load_path}/slot2id.json", "r", encoding="utf-8") as f:
        slot_mapping = json.load(f)

    # Reverse the dictionaries and force the keys to be integers
    intent_unmapping = {int(v): k for k, v in intent_mapping.items()}
    slot_unmapping = {int(v): k for k, v in slot_mapping.items()}
    return word_mapping, slot_unmapping,intent_unmapping

# Execute it in the backend
#creating the dict
saved_word_mapping,  saved_slot_unmapping,saved_intent_unmapping = load_deployment_vocabularies()
#model parameters
EMBEDDING_DIM = 300
HIDDEN_SIZE = 128
NUM_SLOTS = len(saved_slot_unmapping)
NUM_INTENTS = len(saved_intent_unmapping)
VOCAB_SIZE = len(saved_word_mapping)
#model instance
my_sgm_model_instance = SGM(VOCAB_SIZE,EMBEDDING_DIM,HIDDEN_SIZE,NUM_SLOTS,NUM_INTENTS)
#the engine (model) that will be running in the backend for predictions
#we run it in cpu because there is no gradient calculations , so we don't need to use cuda
engine = InferenceEngine(
    model_path="../weights/best_sgm_model.pth",
    model_class=my_sgm_model_instance,
    word2id=saved_word_mapping,
    intent_unmapping=saved_intent_unmapping,
    slot_unmapping=saved_slot_unmapping,
    device="cpu"
)

# API ROUT
# A user sends a POST request
# a simple test
response = engine.predict("I want a  vacation next year.")

print(response)