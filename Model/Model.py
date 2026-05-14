import torch
import torch.nn as nn

class SlotFiling(nn.Module):
    def __init__(self,attin_size):
        super().__init__()
        #replacing cov2d
        self.W_he = nn.Linear(attin_size,attin_size,bias=False)
        self.W_ie = nn.Linear(attin_size,attin_size)
        self.trainable_vector = nn.Parameter(torch.randn(attin_size))
    def forward(self,state_output):
        hidden_feature = self.W_he(state_output)
        hidden_feature = hidden_feature.unsqueeze(1)

        y = self.W_ie(state_output)
        y = y.unsqueeze(2)

        bahdananu_attention = torch.sum(self.trainable_vector * torch.tanh(hidden_feature + y),dim=3)

        learned_attention_weights = torch.softmax(bahdananu_attention,dim= -1 )
        learned_attention_weights = learned_attention_weights.unsqueeze(-1)

        hidden_states = state_output.unsqueeze(1)
        slot_context_vector = torch.sum(learned_attention_weights*hidden_states,dim=2)
        return slot_context_vector

class Intent(nn.Module):
    def __init__(self,attin_size):
        super().__init__()
        #replacing cov2d
        self.W_he = nn.Linear(attin_size,attin_size,bias=False)
        self.W_ie = nn.Linear(attin_size,attin_size)
        self.trainable_vector = nn.Parameter(torch.randn(attin_size))
    def forward(self,final_state,state_output):
        hidden_feature = self.W_he(state_output)

        y = self.W_ie(final_state)
        y = y.unsqueeze(1)

        bahdananu_attention = torch.sum(self.trainable_vector * torch.tanh(hidden_feature + y),dim=-1)

        learned_attention_weights = torch.softmax(bahdananu_attention,dim= -1 )
        learned_attention_weights = learned_attention_weights.unsqueeze(-1)

        intent_context_vector = torch.sum(learned_attention_weights*state_output,dim=1)
        return intent_context_vector

class SlotGate(nn.Module):
    def __init__(self,attin_size,slot_size,intent_size):
        super().__init__()
        self.intent_gate = nn.Linear(attin_size,attin_size)
        self.trainable_vector = nn.Parameter(torch.randn(attin_size))
        self.intent_projection = nn.Linear(attin_size,intent_size)
        self.slot_projection = nn.Linear(attin_size*2,slot_size)
    def forward(self,intent_context_vector,slot_context_vector,state_output):
        intent_gate_output = self.intent_gate(intent_context_vector)
        intent_gate_output = intent_gate_output.unsqueeze(1)
        slot_gate = self.trainable_vector * torch.tanh(slot_context_vector + intent_gate_output)
        slot_gate = torch.sum(slot_gate,-1)
        slot_gate=slot_gate.unsqueeze(-1)
        slot_gate = slot_context_vector * slot_gate
        slot_output = torch.cat([slot_gate,state_output],-1)
        intent = self.intent_projection(intent_context_vector)
        slot = self.slot_projection(slot_output)
        return slot,intent

class SGM(nn.Module):
    def __init__(self,num_embedding,embedding_dim,hidden_size,slot_size,intent_size):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings= num_embedding,embedding_dim=embedding_dim)
        self.bidirectional = nn.GRU(input_size=embedding_dim,hidden_size=hidden_size,bidirectional=True,batch_first=True)
        attin_size = hidden_size *2
        self.slot_filing = SlotFiling(attin_size)
        self.intent = Intent(attin_size)
        self.slot_gate = SlotGate(attin_size,slot_size,intent_size)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self,seq):
        embed_output = self.embed(seq)
        state_output , (final_state) = self.bidirectional(embed_output)
        final_state = self.dropout(final_state)
        state_output = self.dropout(state_output)
        final_state = torch.cat([final_state[0], final_state[1]], dim=-1)
        slot_context_vector = self.slot_filing(state_output)
        intent_context_vector = self.intent(final_state,state_output)
        slot,intent = self.slot_gate(intent_context_vector,slot_context_vector,state_output)
        return  slot,intent
