# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np
import os

# ------------------ Flags ------------------
TRAIN_MODE = False   # set to False if you only want to chat with the saved model

# ------------------ Special tokens ------------------
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"

# ------------------ Model parameters ------------------
d_model = 128
num_heads = 8
num_layers = 4
max_len = 60
batch_size = 16
temperature = 0.8
top_p = 0.92
repetition_penalty = 1.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
    def forward(self, src, tgt,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None,
                tgt_mask=None,
                memory_key_padding_mask=None):
        # src: (B, S)  tgt: (B, T)
        src_emb = self.dropout(self.tok_embed(src) + self.pos_embed[:, :src.size(1), :])
        tgt_emb = self.dropout(self.tok_embed(tgt) + self.pos_embed[:, :tgt.size(1), :])
        
        # pass padding masks to encoder/decoder properly
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        return self.out(output)

# ------------------ Helpers ------------------
def text_to_tensor(words, word2idx):
    idxs = [word2idx.get(w, word2idx[PAD_TOKEN]) for w in words]
    return torch.tensor(idxs, dtype=torch.long, device=device)

def tensor_to_text(tensor, idx2word):
    words = []
    for i in tensor:
        word = idx2word[i.item()]
        if word not in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]:
            words.append(word)
    return " ".join(words)

def create_mask(src, tgt, word2idx):
    """
    Returns: src_padding_mask (B,S) bool, tgt_padding_mask (B,T) bool, tgt_mask (T,T) float causal mask
    """
    tgt_seq_len = tgt.shape[1]
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)  # (T, T)
    
    # Convert boolean masks to float masks with -inf for masked positions
    src_padding_mask = (src == word2idx[PAD_TOKEN]).to(device)   # (B, S)
    tgt_padding_mask = (tgt == word2idx[PAD_TOKEN]).to(device)   # (B, T)
    
    # Convert boolean masks to float masks with -inf for masked positions
    src_padding_mask_float = torch.zeros_like(src, dtype=torch.float32, device=device)
    src_padding_mask_float = src_padding_mask_float.masked_fill(src_padding_mask, float('-inf'))
    
    tgt_padding_mask_float = torch.zeros_like(tgt, dtype=torch.float32, device=device)
    tgt_padding_mask_float = tgt_padding_mask_float.masked_fill(tgt_padding_mask, float('-inf'))
    
    return src_padding_mask_float, tgt_padding_mask_float, tgt_mask

# ------------------ Safe nucleus-sampling util ------------------
def top_p_filter(probs, top_p):
    """
    probs: (1, V) probabilities (after softmax)
    returns filtered probs with entries outside nucleus set to 0, renormalized if possible
    """
    # sort descending
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs > top_p
    # keep at least the top token
    mask[..., 0] = False
    # map mask back to original indices
    indices_to_remove = torch.zeros_like(mask, dtype=torch.bool).scatter(1, sorted_indices, mask)
    filtered = probs.masked_fill(indices_to_remove, 0.0)
    if filtered.sum() == 0:
        # fallback: return original probs (no filtering)
        return probs
    return filtered / filtered.sum()

# ------------------ Response generator (used in training & inference) ------------------
def generate_coherent_response(input_text, model, word2idx, idx2word, max_length=30,
                               temperature_local=temperature, top_p_local=top_p,
                               repetition_penalty_local=repetition_penalty):
    model.eval()
    src = text_to_tensor(input_text.split() + [EOS_TOKEN], word2idx).unsqueeze(0)  # (1, S)
    tgt = torch.tensor([[word2idx[SOS_TOKEN]]], device=device)  # (1,1)
    generated_words, last_tokens = [], []

    with torch.no_grad():
        for _ in range(max_length):
            src_padding_mask, tgt_padding_mask, tgt_mask = create_mask(src, tgt, word2idx)
            output = model(src, tgt, src_key_padding_mask=src_padding_mask,
                           tgt_key_padding_mask=tgt_padding_mask, tgt_mask=tgt_mask)
            next_token_logits = output[:, -1, :]  # (1, V)
            # apply temperature
            next_token_logits = next_token_logits / max(1e-8, temperature_local)

            # apply repetition penalty on logits (safer than dividing probabilities)
            for token_id in set(tgt[0].tolist() + last_tokens[-5:]):
                if token_id != word2idx[SOS_TOKEN]:
                    next_token_logits[0, token_id] /= (repetition_penalty_local * 1.5)

            probs = torch.softmax(next_token_logits, dim=-1)  # (1, V)
            # apply nucleus filtering safely
            probs = top_p_filter(probs, top_p_local)

            # sample or greedy depending on temperature (if temperature very close to 0 -> greedy)
            if temperature_local < 1e-4:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                # ensure sum>0 then sample
                if probs.sum() <= 0:
                    probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)  # (1,1)

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

# ------------------ Training or Inference ------------------
if TRAIN_MODE:
    # training needs Mongo
    from pymongo import MongoClient
    from dotenv import load_dotenv

    load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")
    client = MongoClient(MONGO_URI)
    db = client["lynqbit_db"]
    collection = db["training_data"]

    # Load dataset
    data = list(collection.find())
    questions = [d["question"].lower().split() for d in data]
    answers = [d["answer"].lower().split() for d in data]

    print(f"Loaded {len(questions)} training examples")

    # Vocabulary
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
    print(f"Device: {device} | d_model: {d_model}")

    # Prepare tensors
    tensors_q = [text_to_tensor(q + [EOS_TOKEN], word2idx) for q in questions]
    tensors_a = [text_to_tensor([SOS_TOKEN] + a + [EOS_TOKEN], word2idx) for a in answers]

    q_batch_all = pad_sequence(tensors_q, batch_first=True, padding_value=word2idx[PAD_TOKEN])
    a_batch_all = pad_sequence(tensors_a, batch_first=True, padding_value=word2idx[PAD_TOKEN])

    q_batch_all = q_batch_all[:, :max_len]
    a_batch_all = a_batch_all[:, :max_len]

    # Model, optimizer, loss
    model = ImprovedTransformer(vocab_size, d_model, num_heads, num_layers, max_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx[PAD_TOKEN])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.8)
    epochs = 150
    min_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 12

    print("Starting training with improved transformer...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        # shuffle
        perm = torch.randperm(q_batch_all.size(0))
        q_batch_shuffled = q_batch_all[perm]
        a_batch_shuffled = a_batch_all[perm]

        for i in range(0, len(q_batch_shuffled), batch_size):
            q_batch = q_batch_shuffled[i:i+batch_size].to(device)
            a_batch = a_batch_shuffled[i:i+batch_size].to(device)

            src_padding_mask, tgt_padding_mask, tgt_mask = create_mask(q_batch, a_batch[:, :-1], word2idx)
            optimizer.zero_grad()
            output = model(q_batch, a_batch[:, :-1],
                           src_key_padding_mask=src_padding_mask,
                           tgt_key_padding_mask=tgt_padding_mask,
                           tgt_mask=tgt_mask)  # output: (B, T, V)

            # shift targets: predict a_batch[:,1:]
            loss = criterion(output.reshape(-1, vocab_size), a_batch[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / (num_batches if num_batches > 0 else 1)
        scheduler.step(avg_loss)

        # checkpointing
        if avg_loss < min_loss:
            min_loss = avg_loss
            patience_counter = 0
            torch.save({
                'model_state': model.state_dict(),
                'word2idx': word2idx,
                'idx2word': idx2word,
                'd_model': d_model,
                'num_heads': num_heads,
                'num_layers': num_layers,
                'max_len': max_len
            }, 'best_model.pth')
        else:
            patience_counter += 1

        # monitoring
        if (epoch + 1) % 5 == 0:
            print(f"[Epoch {epoch+1}/{epochs}] avg loss: {avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:  # every 10 epochs show samples
            test_prompts = ["hello", "who created you"]
            print()
            model.eval()
            for prompt in test_prompts:
                reply = generate_coherent_response(prompt, model, word2idx, idx2word, max_length=30, temperature_local=1.0)
                print(f"  '{prompt}' -> '{reply}'")
            print()
            model.train()

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # load best model after training (optional) to be ready for chat within this script
    if os.path.exists('best_model.pth'):
        ckpt = torch.load('best_model.pth', map_location=device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        print("Best model loaded into memory (ready for inference).")

else:
    # ------------------ Inference only ------------------
    if not os.path.exists('best_model.pth'):
        raise FileNotFoundError("best_model.pth not found. Train first or provide checkpoint.")
    checkpoint = torch.load('best_model.pth', map_location=device)
    word2idx = checkpoint['word2idx']
    idx2word = checkpoint['idx2word']
    vocab_size = len(word2idx)

    model = ImprovedTransformer(vocab_size, d_model, num_heads, num_layers, max_len).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print("Loaded model + vocab from best_model.pth")

# ------------------ Chat loop (only runs if TRAIN_MODE is False OR after training you want interactive session) --
if not TRAIN_MODE:
    print("\n😼 Lynqbit is online! Type your message (or !exit to quit)\n")
    while True:
        user_input = input("You: ").strip().lower()
        if user_input == "!exit":
            break
        response = generate_coherent_response(user_input, model, word2idx, idx2word, max_length=30)
        if response:
            emojis = ['😼', '🐾', '⚡', '😹']
            if random.random() > 0.2:
                response += " " + random.choice(emojis)
            print(f"Lynqbit: {response}")
        else:
            print("Lynqbit: *processor purrs* Try asking me something else? 😼")