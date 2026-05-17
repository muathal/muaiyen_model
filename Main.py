from preproccess.Preproccess import *
from Model.Model import SGM
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from seqeval.metrics import f1_score # The academic standard for BIO tags
#to load the on cuda (if possible)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
#setting random state to 42 to get same result every run
torch.manual_seed(42)
#data reading and cleaning
train = pd.read_json("Data/train.json")
val = pd.read_json("Data/val.json")
test = pd.read_json("Data/test.json")
train = clean_entire_dataset(train)
val = clean_entire_dataset(val)
test = clean_entire_dataset(test)
#bulding vocab
word_mapping,word_unmapping = build_vocab(train["text"])
slot_mapping,intent_mapping,slot_unmapping,intent_unmapping = build_mapping(train)
save_vocabularies(word_mapping,slot_mapping,intent_mapping)
#creating the dataset for the model
train_data = ERPDataset(train,word_mapping,slot_mapping,intent_mapping)
val_data = ERPDataset(val,word_mapping,slot_mapping,intent_mapping)
test_data = ERPDataset(test,word_mapping,slot_mapping,intent_mapping)
batch = 32
trainLoader = DataLoader(train_data,shuffle=True,batch_size=batch,collate_fn= collate_fn)
valLoader = DataLoader(val_data,shuffle=False,batch_size=batch,collate_fn=collate_fn)
testLoader = DataLoader(test_data,shuffle=False,batch_size=batch,collate_fn= collate_fn)
# 1.Evaluation Function

def evaluate(model, dataloader, criterion, slot_unmapping, device, pad_token_id=0):
    model.eval()
    total_loss = 0

    all_intent_trues, all_intent_preds = [], []
    all_slot_trues, all_slot_preds = [], []

    with torch.no_grad():
        for texts, batch_slots, intents in dataloader:
            texts = texts.to(device)
            batch_slots = batch_slots.to(device)
            intents = intents.to(device)

            # Forward pass
            slot_logits,intent_logits = model(texts)

            # Loss Calculation (Joint Loss)
            intent_loss = criterion(intent_logits, intents)
            flat_slot_logits = slot_logits.view(-1, slot_logits.shape[-1])
            flat_batch_slots = batch_slots.view(-1)
            slot_loss = criterion(flat_slot_logits, flat_batch_slots)

            total_loss += (intent_loss + slot_loss).item()

            # Get Predictions
            intent_preds = torch.argmax(intent_logits, dim=1)
            slot_preds = torch.argmax(slot_logits, dim=2)

            all_intent_trues.extend(intents.cpu().tolist())
            all_intent_preds.extend(intent_preds.cpu().tolist())

            # Decode Slots (Ignore <PAD> tokens)
            for i in range(texts.size(0)):
                true_slots_seq = []
                pred_slots_seq = []
                for j in range(texts.size(1)):
                    # Only look at real words, ignore padding
                    if texts[i, j].item() != pad_token_id:
                        true_slots_seq.append(slot_unmapping[batch_slots[i, j].item()])
                        pred_slots_seq.append(slot_unmapping[slot_preds[i, j].item()])
                all_slot_trues.append(true_slots_seq)
                all_slot_preds.append(pred_slots_seq)

    #Calculate Metrics
    # 1. Intent Accuracy
    intent_acc = accuracy_score(all_intent_trues, all_intent_preds)

    # 2. Slot F1
    slot_f1 = f1_score(all_slot_trues, all_slot_preds)
    # 3. Semantic Frame Accuracy (Must get Intent AND all Slots right)
    frame_correct = 0
    for i in range(len(all_intent_trues)):
        if (all_intent_trues[i] == all_intent_preds[i]) and (all_slot_trues[i] == all_slot_preds[i]):
            frame_correct += 1
    frame_acc = frame_correct / len(all_intent_trues)


    return total_loss / len(dataloader), intent_acc, slot_f1,frame_acc


#Training Loop

def train_model(model, train_loader, val_loader, optimizer, criterion, slot_unmapping, device, epochs=10, patience=5):
    best_val_frame_acc = 0.0
    epochs_no_improve = 0
    best_val_frame_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        # Setup progress bar
        loop = tqdm(train_loader, leave=True)
        loop.set_description(f"Epoch {epoch+1}/{epochs}")

        for texts, batch_slots, intents in loop:
            texts = texts.to(device)
            batch_slots = batch_slots.to(device)
            intents = intents.to(device)

            optimizer.zero_grad()

            # Forward
            slot_logits,intent_logits = model(texts)

            # Joint Loss
            intent_loss = criterion(intent_logits, intents)
            slot_loss = criterion(slot_logits.view(-1, slot_logits.shape[-1]), batch_slots.view(-1))
            loss = intent_loss + slot_loss

            # Backward
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Validation Phase
        val_loss, val_intent_acc, val_slot_f1, val_frame_acc = evaluate(
            model, val_loader, criterion, slot_unmapping, device
        )

        print(f"\nepoch {epoch+1} Results:")
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
        print(f"Intent Acc: {val_intent_acc:.4f} | Slot F1: {val_slot_f1:.4f} | Frame Acc: {val_frame_acc:.4f}")

        # 3. The Early Stopping Logic
        if val_frame_acc > best_val_frame_acc:
            best_val_frame_acc = val_frame_acc
            torch.save(model.state_dict(), "weights/best_sgm_model.pth")
            print("New best model saved!\n")

            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).\n")

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered! No improvement for {patience} epochs.")
                print("stopping training to prevent overfitting and save compute.")
                break

# 3. Execution
EMBEDDING_DIM = 300
HIDDEN_SIZE = 128
NUM_SLOTS = len(slot_mapping)
NUM_INTENTS = len(intent_mapping)
VOCAB_SIZE = len(word_mapping)
model = SGM(VOCAB_SIZE,EMBEDDING_DIM,HIDDEN_SIZE,NUM_SLOTS,NUM_INTENTS).to(device)
model.to(device)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0) # Ignore <PAD>
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Starting Training")
train_model(model, trainLoader, valLoader, optimizer, criterion, slot_unmapping, device, epochs=300)

print("Running Final Test Set")
model.load_state_dict(torch.load("weights/best_sgm_model.pth"))
test_loss, test_intent_acc, test_slot_f1, test_frame_acc = evaluate(model, testLoader, criterion, slot_unmapping, device)

print(f"FINAL TEST RESULTS")
print(f"Intent Accuracy: {test_intent_acc:.4f}")
print(f"Slot F1 Score:   {test_slot_f1:.4f}")
print(f"Frame Accuracy:  {test_frame_acc:.4f}")