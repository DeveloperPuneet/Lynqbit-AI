# main.py
import torch # Import PyTorch 🤖
import torch.nn as nn # Neural network module 🧠
import torch.optim as optim # For optimization ⚙️
from torch.nn.utils.rnn import pad_sequence # Pad sequences 📏
import random # For randomness 🎲
import numpy as np # For numerical ops 🧮
import os # For OS interaction 💻

# ------------------ Flags ------------------
TRAIN_MODE = False   # set to False if you only want to chat with the saved model

# ------------------ Special tokens ------------------
SOS_TOKEN = "<SOS>" # Start token 🏁
EOS_TOKEN = "<EOS>" # End token 🏁
PAD_TOKEN = "<PAD>" # Padding token 🧱

# ------------------ Model parameters ------------------
d_model = 256 # Embedding dimension 📏
num_heads = 8 # Attention heads count 🧠
num_layers = 4 # Transformer layers count 🧱
max_len = 60 # Maximum sequence length 📏
batch_size = 16 # Batch size 📦
temperature = 0.95 # Sampling temperature 🌡️
top_p = 0.92 # Nucleus sampling parameter ☢️
repetition_penalty = 1.2 # Repetition penalty factor ⚖️
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available 🚀

# ------------------ Enhanced Transformer Model ------------------
class ImprovedTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads=8, num_layers=4, max_len=60, dropout=0.1):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model) # Token embeddings 📝
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model)) # Positional embeddings 📍
        self.dropout = nn.Dropout(dropout) # Dropout layer 💧
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) # Encoder layer 🧱
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=dropout
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers) # Decoder layer 🧱
        
        self.out = nn.Linear(d_model, vocab_size) # Output layer 📤
        
    def forward(self, src, tgt,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None,
                tgt_mask=None,
                memory_key_padding_mask=None):
        # src: (B, S)  tgt: (B, T)
        src_emb = self.dropout(self.tok_embed(src) + self.pos_embed[:, :src.size(1), :]) # Embed source ✍️
        tgt_emb = self.dropout(self.tok_embed(tgt) + self.pos_embed[:, :tgt.size(1), :]) # Embed target ✍️
        
        # pass padding masks to encoder/decoder properly
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask) # Encode source 🧱
        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        ) # Decode target 🧱
        return self.out(output) # Output logits 📤

# ------------------ Helpers ------------------
def text_to_tensor(words, word2idx):
    idxs = [word2idx.get(w, word2idx[PAD_TOKEN]) for w in words] # Get indices 🔢
    return torch.tensor(idxs, dtype=torch.long, device=device) # Convert to tensor ✍️

def tensor_to_text(tensor, idx2word):
    words = [] # Initialize words list 📝
    for i in tensor: # Iterate tensor elements 🔄
        word = idx2word[i.item()] # Get word 💬
        if word not in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]: # Filter special tokens 🚫
            words.append(word) # Add word to list ✅
    return " ".join(words) # Join words with space 🗣️

def create_mask(src, tgt, word2idx):
    """
    Returns: src_padding_mask (B,S) bool, tgt_padding_mask (B,T) bool, tgt_mask (T,T) float causal mask
    """
    tgt_seq_len = tgt.shape[1] # Target sequence length 📏
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)  # (T, T) generate mask 🎭
    
    # Convert boolean masks to float masks with -inf for masked positions
    src_padding_mask = (src == word2idx[PAD_TOKEN]).to(device)   # (B, S) Create source mask 🎭
    tgt_padding_mask = (tgt == word2idx[PAD_TOKEN]).to(device)   # (B, T) Create target mask 🎭
    
    # Convert boolean masks to float masks with -inf for masked positions
    src_padding_mask_float = torch.zeros_like(src, dtype=torch.float32, device=device) # Create src mask float 🎭
    src_padding_mask_float = src_padding_mask_float.masked_fill(src_padding_mask, float('-inf')) # Fill mask 🎭
    
    tgt_padding_mask_float = torch.zeros_like(tgt, dtype=torch.float32, device=device) # Create tgt mask float 🎭
    tgt_padding_mask_float = tgt_padding_mask_float.masked_fill(tgt_padding_mask, float('-inf')) # Fill mask 🎭
    
    return src_padding_mask_float, tgt_padding_mask_float, tgt_mask # Return masks 🎭

# ------------------ Safe nucleus-sampling util ------------------
def top_p_filter(probs, top_p):
    """
    probs: (1, V) probabilities (after softmax)
    returns filtered probs with entries outside nucleus set to 0, renormalized if possible
    """
    # sort descending
    sorted_probs, sorted_indices = torch.sort(probs, descending=True) # Sort probs ⬆️
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1) # Cumulative probs ➕
    mask = cumulative_probs > top_p # Create mask 🎭
    # keep at least the top token
    mask[..., 0] = False # Always keep top one ✅
    # map mask back to original indices
    indices_to_remove = torch.zeros_like(mask, dtype=torch.bool).scatter(1, sorted_indices, mask) # Get indices 🔢
    filtered = probs.masked_fill(indices_to_remove, 0.0) # Apply mask 🎭
    if filtered.sum() == 0:
        # fallback: return original probs (no filtering)
        return probs # Return original probs 🔄
    return filtered / filtered.sum() # Normalize filtered probs ⚖️

# ------------------ Response generator (used in training & inference) ------------------
def generate_coherent_response(input_text, model, word2idx, idx2word, max_length=30,
                               temperature_local=temperature, top_p_local=top_p,
                               repetition_penalty_local=repetition_penalty):
    model.eval() # Set to eval mode 🚦
    src = text_to_tensor(input_text.split() + [EOS_TOKEN], word2idx).unsqueeze(0)  # (1, S) Prepare source 📝
    tgt = torch.tensor([[word2idx[SOS_TOKEN]]], device=device)  # (1,1) Prepare target 📝
    generated_words, last_tokens = [], [] # Initialize lists 📝

    with torch.no_grad(): # Disable gradient calculation 🚫
        for _ in range(max_length): # Generate tokens loop 🔄
            src_padding_mask, tgt_padding_mask, tgt_mask = create_mask(src, tgt, word2idx) # Create masks 🎭
            output = model(src, tgt, src_key_padding_mask=src_padding_mask,
                           tgt_key_padding_mask=tgt_padding_mask, tgt_mask=tgt_mask) # Get model output 📤
            next_token_logits = output[:, -1, :]  # (1, V) Get next logits 🔢
            # apply temperature
            next_token_logits = next_token_logits / max(1e-8, temperature_local) # Apply temperature 🌡️

            # apply repetition penalty on logits (safer than dividing probabilities)
            for token_id in set(tgt[0].tolist() + last_tokens[-5:]): # Iterate tokens 🔄
                if token_id != word2idx[SOS_TOKEN]: # Filter SOS token 🚫
                    next_token_logits[0, token_id] /= (repetition_penalty_local * 1.5) # Apply penalty ⚖️

            probs = torch.softmax(next_token_logits, dim=-1)  # (1, V) Softmax probs 💡
            # apply nucleus filtering safely
            probs = top_p_filter(probs, top_p_local) # Apply top-p filtering ☢️

            # sample or greedy depending on temperature (if temperature very close to 0 -> greedy)
            if temperature_local < 1e-4: # Check temp value 🌡️
                next_token = torch.argmax(probs, dim=-1, keepdim=True) # Greedy selection 🥇
            else:
                # ensure sum>0 then sample
                if probs.sum() <= 0: # Check prob sum ➕
                    probs = torch.softmax(next_token_logits, dim=-1) # Softmax probs 💡
                next_token = torch.multinomial(probs, 1)  # (1,1) Sample token 🎲

            next_word = idx2word[next_token.item()] # Get next word 💬
            if (next_word in [EOS_TOKEN, PAD_TOKEN] or 
                (len(generated_words) > 3 and next_word in generated_words[-3:])): # Check tokens 🚫
                break # Exit loop 🛑

            tgt = torch.cat([tgt, next_token], dim=1) # Concatenate target ➕
            generated_words.append(next_word) # Add word to list ✅
            last_tokens.append(next_token.item()) # Add token ID 🔢

            if next_token.item() == word2idx[EOS_TOKEN]: # Check EOS token 🏁
                break # Exit loop 🛑

    return " ".join(generated_words) # Join words with space 🗣️

# ------------------ Training or Inference ------------------
if TRAIN_MODE: # Training mode on? ✅
    # training needs Mongo
    from pymongo import MongoClient # Import mongo client 💾
    from dotenv import load_dotenv # Load environment variables ⚙️

    load_dotenv() # Load env vars ⚙️
    MONGO_URI = os.getenv("MONGO_URI") # Get Mongo URI 🔗
    client = MongoClient(MONGO_URI) # Create mongo client 💾
    db = client["lynqbit_db"] # Get database 💾
    collection = db["training_data"] # Get collection 💾

    # Load dataset
    data = list(collection.find()) # Load data from DB 💾
    questions = [d["question"].lower().split() for d in data] # Get questions ❓
    answers = [d["answer"].lower().split() for d in data] # Get answers 💬

    print(f"Loaded {len(questions)} training examples") # Print data info ℹ️

    # Vocabulary
    all_words = set() # Create word set 📝
    for q in questions: # Iterate questions 🔄
        all_words.update(q) # Add words to set ✅
    for a in answers: # Iterate answers 🔄
        all_words.update(a) # Add words to set ✅
    all_words.update([SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]) # Add special tokens ✅

    word2idx = {w: i for i, w in enumerate(sorted(all_words))} # Word to index mapping 🔢
    idx2word = {i: w for w, i in word2idx.items()} # Index to word mapping 💬
    vocab_size = len(word2idx) # Get vocab size 📏

    print(f"Vocabulary size: {vocab_size}") # Print vocab size ℹ️
    print(f"Device: {device} | d_model: {d_model}") # Print device info ℹ️

    # Prepare tensors
    tensors_q = [text_to_tensor(q + [EOS_TOKEN], word2idx) for q in questions] # Create question tensors ✍️
    tensors_a = [text_to_tensor([SOS_TOKEN] + a + [EOS_TOKEN], word2idx) for a in answers] # Create answer tensors ✍️

    q_batch_all = pad_sequence(tensors_q, batch_first=True, padding_value=word2idx[PAD_TOKEN]) # Pad questions 📏
    a_batch_all = pad_sequence(tensors_a, batch_first=True, padding_value=word2idx[PAD_TOKEN]) # Pad answers 📏

    q_batch_all = q_batch_all[:, :max_len] # Truncate questions ✂️
    a_batch_all = a_batch_all[:, :max_len] # Truncate answers ✂️

    # Model, optimizer, loss
    model = ImprovedTransformer(vocab_size, d_model, num_heads, num_layers, max_len).to(device) # Initialize model 🤖
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5) # Adam optimizer ⚙️
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx[PAD_TOKEN]) # Cross entropy loss 📉
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.8) # LR scheduler ⏱️
    epochs = 150 # Training epochs ⏳
    min_loss = float('inf') # Init min loss ♾️
    patience_counter = 0 # Init patience counter ⏳
    early_stopping_patience = 12 # Early stopping patience ⏳

    print("Starting training with improved transformer...") # Print start message ℹ️

    for epoch in range(epochs): # Training loop ⏳
        model.train() # Set to train mode 🚦
        total_loss = 0.0 # Initialize loss 📉
        num_batches = 0 # Initialize batches count 🔢
        # shuffle
        perm = torch.randperm(q_batch_all.size(0)) # Shuffle indices 🎲
        q_batch_shuffled = q_batch_all[perm] # Shuffle questions 📦
        a_batch_shuffled = a_batch_all[perm] # Shuffle answers 📦

        for i in range(0, len(q_batch_shuffled), batch_size): # Batch loop 📦
            q_batch = q_batch_shuffled[i:i+batch_size].to(device) # Get batch 📦
            a_batch = a_batch_shuffled[i:i+batch_size].to(device) # Get batch 📦

            src_padding_mask, tgt_padding_mask, tgt_mask = create_mask(q_batch, a_batch[:, :-1], word2idx) # Create masks 🎭
            optimizer.zero_grad() # Zero gradients 📉
            output = model(q_batch, a_batch[:, :-1],
                           src_key_padding_mask=src_padding_mask,
                           tgt_key_padding_mask=tgt_padding_mask,
                           tgt_mask=tgt_mask)  # output: (B, T, V) Get model output 📤

            # shift targets: predict a_batch[:,1:]
            loss = criterion(output.reshape(-1, vocab_size), a_batch[:, 1:].reshape(-1)) # Calculate loss 📉
            loss.backward() # Backpropagate loss ↩️
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) # Clip gradients ✂️
            optimizer.step() # Update weights ⬆️

            total_loss += loss.item() # Accumulate loss 📉
            num_batches += 1 # Increment batch count 🔢

        avg_loss = total_loss / (num_batches if num_batches > 0 else 1) # Calculate avg loss ➗
        scheduler.step(avg_loss) # Step scheduler ⏱️

        # checkpointing
        if avg_loss < min_loss: # Check loss value 📉
            min_loss = avg_loss # Update min loss 📉
            patience_counter = 0 # Reset patience counter ⏳
            torch.save({ # Save model state 💾
                'model_state': model.state_dict(), # Save model state 💾
                'word2idx': word2idx, # Save word2idx mapping 🔢
                'idx2word': idx2word, # Save idx2word mapping 💬
                'd_model': d_model, # Save d_model param 📏
                'num_heads': num_heads, # Save num_heads param 🧠
                'num_layers': num_layers, # Save num_layers param 🧱
                'max_len': max_len # Save max_len param 📏
            }, 'best_model.pth')
        else:
            patience_counter += 1 # Increment patience counter ⏳

        # monitoring
        if (epoch + 1) % 5 == 0: # Print every 5 epochs 🗓️
            print(f"[Epoch {epoch+1}/{epochs}] avg loss: {avg_loss:.4f}") # Print loss value 📉

        if (epoch + 1) % 10 == 0:  # every 10 epochs show samples
            test_prompts = ["hello", "who created you"] # Test prompts 💬
            print() # Print empty line ⚪
            model.eval() # Set to eval mode 🚦
            for prompt in test_prompts: # Iterate prompts 🔄
                reply = generate_coherent_response(prompt, model, word2idx, idx2word, max_length=30, temperature_local=1.0) # Generate reply 🗣️
                print(f"  '{prompt}' -> '{reply}'") # Print prompt and reply 💬
            print() # Print empty line ⚪
            model.train() # Set to train mode 🚦

        if patience_counter >= early_stopping_patience: # Check patience count ⏳
            print(f"Early stopping at epoch {epoch+1}") # Print stop message 🛑
            break # Exit loop 🛑

    # load best model after training (optional) to be ready for chat within this script
    if os.path.exists('best_model.pth'): # Check file exists 📁
        ckpt = torch.load('best_model.pth', map_location=device) # Load checkpoint 💾
        model.load_state_dict(ckpt['model_state']) # Load model state 🤖
        model.eval() # Set to eval mode 🚦
        print("Best model loaded into memory (ready for inference).") # Print load message ℹ️

else:
    # ------------------ Inference only ------------------
    if not os.path.exists('best_model.pth'): # Check file exists 📁
        raise FileNotFoundError("best_model.pth not found. Train first or provide checkpoint.") # Raise error ❌
    checkpoint = torch.load('best_model.pth', map_location=device) # Load checkpoint 💾
    word2idx = checkpoint['word2idx'] # Load word2idx mapping 🔢
    idx2word = checkpoint['idx2word'] # Load idx2word mapping 💬
    vocab_size = len(word2idx) # Get vocab size 📏

    model = ImprovedTransformer(vocab_size, d_model, num_heads, num_layers, max_len).to(device) # Initialize model 🤖
    model.load_state_dict(checkpoint['model_state']) # Load model state 🤖
    model.eval() # Set to eval mode 🚦
    print("Loaded model + vocab from best_model.pth") # Print load message ℹ️

# ------------------ Chat loop (only runs if TRAIN_MODE is False OR after training you want interactive session) --
if not TRAIN_MODE: # Check train mode 🚦
    print("\n😼 Lynqbit is online! Type your message (or !exit to quit)\n") # Print welcome message 😼
    while True: # Chat loop 💬
        user_input = input("You: ").strip().lower() # Get user input 🗣️
        if user_input == "!exit": # Check exit command 🚪
            break # Exit loop 🛑
        response = generate_coherent_response(user_input, model, word2idx, idx2word, max_length=30) # Generate response 🗣️
        if response: # Check response 💬
            emojis = ['😼', '🐾', '⚡', '😹'] # Emojis list 🤩
            if random.random() > 0.2: # Random condition 🎲
                response += " " + random.choice(emojis) # Add emoji ✅
            print(f"Lynqbit: {response}") # Print response 💬
        else:
            print("Lynqbit: *processor purrs* Try asking me something else? 😼") # Print error message 😼