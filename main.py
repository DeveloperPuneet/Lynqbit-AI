# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np

# ------------------ Load environment ------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["lynqbit_db"]
collection = db["training_data"]

# ------------------ Load dataset ------------------
data = list(collection.find())
questions = [d["question"].lower().split() for d in data]
answers = [d["answer"].lower().split() for d in data]

print(f"Loaded {len(questions)} training examples")

# ------------------ Special tokens ------------------
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"

# ------------------ Vocabulary ------------------
all_words = set()
for q in questions:
    all_words.update(q)
for a in answers:
    all_words.update(a)
all_words.update([SOS_TOKEN, EOS_TOKEN, PAD_TOKEN])

word2idx = {w: i for i, w in enumerate(sorted(all_words))}
idx2word = {i: w for w, i in word2idx.items()}

vocab_size = len(word2idx)
print(f"Vocabulary size: {vocab_size}")

# Model parameters
d_model = 128
num_heads = 8
num_layers = 4
max_len = 60
batch_size = 16
temperature = 0.8
top_p = 0.92
repetition_penalty = 1.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | d_model: {d_model}")

# ------------------ Enhanced Transformer Model ------------------
class ImprovedTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads=8, num_layers=4, max_len=60, dropout=0.1):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=dropout
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.out = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.dropout(self.tok_embed(src) + self.pos_embed[:, :src.size(1), :])
        tgt_emb = self.dropout(self.tok_embed(tgt) + self.pos_embed[:, :tgt.size(1), :])
        
        memory = self.encoder(src_emb)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        
        return self.out(output)

# ------------------ Helpers ------------------
def text_to_tensor(words):
    idxs = [word2idx.get(w, word2idx[PAD_TOKEN]) for w in words]
    return torch.tensor(idxs, dtype=torch.long, device=device)

def tensor_to_text(tensor):
    words = []
    for i in tensor:
        word = idx2word[i.item()]
        if word not in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]:
            words.append(word)
    return " ".join(words)

def create_mask(src, tgt):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]
    
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)
    src_padding_mask = (src == word2idx[PAD_TOKEN])
    tgt_padding_mask = (tgt == word2idx[PAD_TOKEN])
    
    return src_padding_mask, tgt_padding_mask, tgt_mask

# ------------------ Prepare tensors ------------------
tensors_q = [text_to_tensor(q + [EOS_TOKEN]) for q in questions]
tensors_a = [text_to_tensor([SOS_TOKEN] + a + [EOS_TOKEN]) for a in answers]

# Pad sequences
q_batch_all = pad_sequence(tensors_q, batch_first=True, padding_value=word2idx[PAD_TOKEN])
a_batch_all = pad_sequence(tensors_a, batch_first=True, padding_value=word2idx[PAD_TOKEN])

# Trim to max_len
q_batch_all = q_batch_all[:, :max_len]
a_batch_all = a_batch_all[:, :max_len]

# ------------------ Training ------------------
model = ImprovedTransformer(vocab_size, d_model, num_heads, num_layers, max_len).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(ignore_index=word2idx[PAD_TOKEN])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.8)
epochs = 150  # Increased epochs as requested

# Training monitoring
min_loss = float('inf')
best_model_state = None
early_stopping_patience = 12  # Increased patience for more epochs
patience_counter = 0

print("Starting training with improved transformer...")

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Shuffle the data
    indices = torch.randperm(len(q_batch_all))
    q_batch_all = q_batch_all[indices]
    a_batch_all = a_batch_all[indices]
    
    for i in range(0, len(q_batch_all), batch_size):
        q_batch = q_batch_all[i:i+batch_size].to(device)
        a_batch = a_batch_all[i:i+batch_size].to(device)
        
        # Create masks
        src_padding_mask, tgt_padding_mask, tgt_mask = create_mask(q_batch, a_batch[:, :-1])
        
        optimizer.zero_grad()
        output = model(q_batch, a_batch[:, :-1], src_padding_mask, tgt_mask)
        
        loss = criterion(output.reshape(-1, vocab_size), a_batch[:, 1:].reshape(-1))
        loss.backward()
        
        # Gradient clipping with smaller norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    scheduler.step(avg_loss)
    
    # Early stopping and model checkpointing
    if avg_loss < min_loss:
        min_loss = avg_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
    
    if (epoch + 1) % 5 == 0:
        print(f"[Epoch {epoch+1}/{epochs}] avg loss: {avg_loss:.4f}")
        
        # Generate ONLY 2 samples as requested
        model.eval()
        with torch.no_grad():
            # Only 2 sample inputs as requested
            test_inputs = ["hello", "who created you"]  # Reduced from 4 to 2
            
            for test_input in test_inputs:
                words = test_input.split()
                src = text_to_tensor(words + [EOS_TOKEN]).unsqueeze(0)
                tgt = torch.tensor([[word2idx[SOS_TOKEN]]], device=device)
                
                for _ in range(30):
                    src_padding_mask, tgt_padding_mask, tgt_mask = create_mask(src, tgt)
                    output = model(src, tgt, src_padding_mask, tgt_mask)
                    next_token_logits = output[:, -1, :]
                    
                    # Apply temperature and repetition penalty
                    next_token_logits = next_token_logits / temperature
                    
                    # Apply repetition penalty
                    for token in set(tgt[0].tolist()):
                        if token != word2idx[SOS_TOKEN]:
                            next_token_logits[0, token] /= repetition_penalty
                    
                    probs = torch.softmax(next_token_logits, dim=-1)
                    
                    # Apply nucleus sampling (top-p)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    probs = probs.masked_fill(indices_to_remove, 0.0)
                    
                    if probs.sum() > 0:
                        probs = probs / probs.sum()
                        next_token = torch.multinomial(probs, 1)
                    else:
                        next_token = torch.argmax(probs, dim=-1, keepdim=True)
                    
                    tgt = torch.cat([tgt, next_token], dim=1)
                    
                    if next_token.item() == word2idx[EOS_TOKEN] or tgt.size(1) >= max_len:
                        break
                
                response = tensor_to_text(tgt[0])
                print(f"  '{test_input}' -> '{response}'")
        
        model.train()
    
    # Early stopping
    if patience_counter >= early_stopping_patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

# ------------------ Enhanced Response Generation ------------------
def generate_coherent_response(input_text, max_length=30):
    words = input_text.split()
    src = text_to_tensor(words + [EOS_TOKEN]).unsqueeze(0)
    tgt = torch.tensor([[word2idx[SOS_TOKEN]]], device=device)
    
    generated_words = []
    last_tokens = []
    
    with torch.no_grad():
        for _ in range(max_length):
            src_padding_mask, tgt_padding_mask, tgt_mask = create_mask(src, tgt)
            output = model(src, tgt, src_padding_mask, tgt_mask)
            next_token_logits = output[:, -1, :]
            
            # Apply temperature and repetition penalty
            next_token_logits = next_token_logits / temperature
            
            # Stronger repetition penalty
            for token in set(tgt[0].tolist() + last_tokens[-5:]):
                if token != word2idx[SOS_TOKEN]:
                    next_token_logits[0, token] /= repetition_penalty * 1.5
            
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Apply nucleus sampling (top-p)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs = probs.masked_fill(indices_to_remove, 0.0)
            
            if probs.sum() > 0:
                probs = probs / probs.sum()
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            
            # Check for coherence - don't allow nonsense sequences
            next_word = idx2word[next_token.item()]
            if (next_word in [EOS_TOKEN, PAD_TOKEN] or 
                (len(generated_words) > 3 and next_word in generated_words[-3:])):
                break
            
            tgt = torch.cat([tgt, next_token], dim=1)
            generated_words.append(next_word)
            last_tokens.append(next_token.item())
            
            if next_token.item() == word2idx[EOS_TOKEN]:
                break
    
    return " ".join(generated_words)

# ------------------ Enhanced Chat Loop ------------------
model.eval()
print("\n😼 Lynqbit is online! Type your message (or !exit to quit)\n")

while True:
    user_input = input("You: ").strip().lower()
    if user_input == "!exit":
        break
    
    # Use the enhanced response generator
    response = generate_coherent_response(user_input)
    
    # Post-processing for better responses
    if response:
        # Add Lynqbit's personality
        emojis = ['😼', '🐾', '⚡', '😹']
        if random.random() > 0.2:
            response += " " + random.choice(emojis)
        
        # Ensure response isn't empty
        if len(response.strip()) < 3:
            fallback_responses = [
                "Purrr... I'm still learning! Ask me something else? 😼",
                "Meow... let me think about that one! 😴",
                "Hiss... I need more training for that question! 🐾"
            ]
            response = random.choice(fallback_responses)
        
        print(f"Lynqbit: {response}")
    else:
        print("Lynqbit: *processor purrs* Try asking me something else? 😼")

## can shift to last code good 