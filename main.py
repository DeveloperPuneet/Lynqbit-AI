# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np
import os
import re
from collections import Counter
import csv
from datetime import datetime
import json
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing
from torch.utils.data import Dataset, DataLoader, random_split
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import requests
from bs4 import BeautifulSoup

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')  

# ------------------ Ephemeral session memory ------------------
session_memory = []  # stores (user_input, lynqbit_response)
MAX_SESSION_TURNS = 50

# ------------------ BPE Tokenizer ------------------
VOCAB_SIZE = 10000  # Target vocabulary size for BPE
# Robust global variable initialization
global bpe_tokenizer, word2idx, idx2word, model, optimizer, vocab_size
bpe_tokenizer = None
word2idx = None
idx2word = None
model = None
optimizer = None
vocab_size = None

# ------------------ Personality/Style Tokens ------------------
PERSONALITY_TOKENS = {
    "<PLAYFUL>": "😺",
    "<MYSTERIOUS>": "🔮", 
    "<FORMAL>": "🎩",
    "<CASUAL>": "👕",
    "<EXCITED>": "🎉",
    "<CALM>": "🍃"
}

# Default personality
CURRENT_PERSONALITY = "<PLAYFUL>"

# ------------------ Intent Tokens ------------------
INTENT_TOKENS = {
    "<QUESTION>": "❓",
    "<STATEMENT>": "💬",
    "<COMMAND>": "⚡",
    "<GREETING>": "👋",
    "<FAREWELL>": "👋"
}

# ------------------ Flags ------------------
TRAIN_MODE = True
ENGLISH_MODEL_TRAIN = False  # Set to False to skip English model pretraining
DEBUG_THINKING = True
MINI_REPLAY_ENABLED = True
LOGGING_ENABLED = True
USE_BPE_TOKENIZATION = True  # Switch between BPE and word-level tokenization

# ------------------ Special tokens ------------------
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>" 
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# Add personality tokens to special tokens
SPECIAL_TOKENS = [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN] + list(PERSONALITY_TOKENS.keys()) + list(INTENT_TOKENS.keys())

# ------------------ Model parameters ------------------
d_model = 256
num_heads = 8
num_layers = 4
max_len = 256  # Increased from 60 to 256 for better context
batch_size = 16
temperature = 1.0
top_p = 0.98
repetition_penalty = 1.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dropout_rate = 0.3
word_dropout_rate = 0.1

# ------------------ Multi-turn Conversation Dataset ------------------
class MultiTurnDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_turns=3):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        
        # Take the last max_turns exchanges
        recent_exchanges = conversation[-self.max_turns:]
        
        # Build context from previous turns
        context_parts = []
        for exchange in recent_exchanges[:-1]:
            context_parts.append(f"user: {exchange['user']}")
            context_parts.append(f"assistant: {exchange['assistant']}")
        
        context = " ".join(context_parts)
        target = recent_exchanges[-1]["assistant"]
        
        return context, target

class PretrainDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=256):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Split text into sentences for better training
        sentences = re.split(r'(?<=[.!?]) +', text)
        self.samples = []
        
        for i in range(len(sentences) - 1):
            # Create source-target pairs from consecutive sentences
            source = sentences[i]
            target = sentences[i + 1]
            
            if len(source.split()) > 3 and len(target.split()) > 3:  # Filter very short sentences
                source_tokens = tokenizer.encode(source).ids
                target_tokens = tokenizer.encode(target).ids
                
                # Limit length
                source_tokens = source_tokens[:block_size//2]
                target_tokens = target_tokens[:block_size//2]
                
                if source_tokens and target_tokens:
                    self.samples.append((source_tokens, target_tokens))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        source, target = self.samples[idx]
        return torch.tensor(source), torch.tensor(target)

# ------------------ Custom Collate Function for Pretraining ------------------
def pretrain_collate_fn(batch):
    """Custom collate function for pretraining dataset"""
    sources, targets = zip(*batch)
    
    # Pad sequences
    sources_padded = pad_sequence(sources, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    
    return sources_padded, targets_padded

# ------------------ Unified Transformer Model ------------------
class ImprovedTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads=8, num_layers=4, max_len=256, dropout=0.1, padding_idx=None):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=dropout,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.out = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src,
        tgt,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        tgt_mask=None,
        memory_key_padding_mask=None,
    ):
        src_emb = self.dropout(self.tok_embed(src) + self.pos_embed[:, : src.size(1), :])
        tgt_emb = self.dropout(self.tok_embed(tgt) + self.pos_embed[:, : tgt.size(1), :])

        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.out(output)

def pretrain_model(file_path="english_model.txt", epochs=5):
    print(">>> Starting Pretraining on English corpus...")
    # Train tokenizer if not already done
    global bpe_tokenizer
    if bpe_tokenizer is None:
        bpe_tokenizer = ByteLevelBPETokenizer()
        bpe_tokenizer.train(files=file_path, vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=SPECIAL_TOKENS)
        bpe_tokenizer.save_model("tokenizer")

    dataset = PretrainDataset(file_path, bpe_tokenizer, block_size=128)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pretrain_collate_fn)

    vocab_size = bpe_tokenizer.get_vocab_size()
    padding_idx = bpe_tokenizer.token_to_id(PAD_TOKEN)
    model = ImprovedTransformer(vocab_size, d_model=d_model, num_heads=num_heads, 
                               num_layers=num_layers, max_len=max_len, 
                               dropout=dropout_rate, padding_idx=padding_idx).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            
            # Add SOS and EOS tokens
            sos_tensor = torch.full((src.size(0), 1), bpe_tokenizer.token_to_id(SOS_TOKEN), device=device)
            eos_tensor = torch.full((tgt.size(0), 1), bpe_tokenizer.token_to_id(EOS_TOKEN), device=device)
            
            src = torch.cat([sos_tensor, src, eos_tensor], dim=1)
            tgt_input = torch.cat([sos_tensor, tgt], dim=1)
            tgt_output = torch.cat([tgt, eos_tensor], dim=1)
            
            # Pad sequences
            src = pad_sequence([s for s in src], batch_first=True, padding_value=padding_idx)
            tgt_input = pad_sequence([t for t in tgt_input], batch_first=True, padding_value=padding_idx)
            tgt_output = pad_sequence([t for t in tgt_output], batch_first=True, padding_value=padding_idx)
            
            # Truncate to max_len
            src = src[:, :max_len]
            tgt_input = tgt_input[:, :max_len]
            tgt_output = tgt_output[:, :max_len]
            
            # Create masks
            src_padding_mask = (src == padding_idx).to(device)
            tgt_padding_mask = (tgt_input == padding_idx).to(device)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            
            optimizer.zero_grad()
            output = model(
                src,
                tgt_input,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                tgt_mask=tgt_mask,
            )
            
            loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[Pretrain] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Sample from the model during pretraining
        if epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                test_prompts = [
                    "What is",
                    "The weather is",
                    "I like to",
                    "Hello, how are you",
                    "Can you tell me about"
                ]
                
                print("\n📝 Pretraining Samples:")
                for prompt in test_prompts:
                    tokens = text_to_bpe_tokens(prompt, add_special_tokens=False)
                    src = torch.tensor([tokens], device=device)
                    
                    # Generate response
                    response = generate_response(src, model, bpe_tokenizer, max_length=20)
                    print(f"  '{prompt}' -> '{response}'")
                print()
            
            model.train()

    torch.save(model.state_dict(), "pretrained_model.pt")
    print(">>> Pretraining finished. Model saved as pretrained_model.pt")
    return model, bpe_tokenizer

# ------------------ BPE Tokenizer Functions ------------------
def train_bpe_tokenizer(texts, vocab_size=VOCAB_SIZE):
    """Train a BPE tokenizer on the provided texts"""
    global bpe_tokenizer
    
    # Initialize a tokenizer
    bpe_tokenizer = ByteLevelBPETokenizer()
    
    # Prepare a temporary file with all texts
    temp_file = "temp_training_text.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")
    
    # Train the tokenizer
    bpe_tokenizer.train(
        files=[temp_file],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS
    )
    
    # Set post-processing template
    bpe_tokenizer.post_processor = TemplateProcessing(
        single=f"{SOS_TOKEN} $A {EOS_TOKEN}",
        pair=f"{SOS_TOKEN} $A {EOS_TOKEN} $B {EOS_TOKEN}",
        special_tokens=[(SOS_TOKEN, bpe_tokenizer.token_to_id(SOS_TOKEN)), 
                       (EOS_TOKEN, bpe_tokenizer.token_to_id(EOS_TOKEN))],
    )
    
    # Clean up
    os.remove(temp_file)
    
    # Save tokenizer
    os.makedirs("tokenizer", exist_ok=True)
    bpe_tokenizer.save_model("tokenizer")
    bpe_tokenizer.save("tokenizer/tokenizer.json")
    
    return bpe_tokenizer

def load_bpe_tokenizer():
    """Load a pre-trained BPE tokenizer"""
    global bpe_tokenizer
    
    # Check if tokenizer files exist
    vocab_path = "tokenizer/vocab.json"
    merges_path = "tokenizer/merges.txt"
    
    # If files are empty or don't exist, return None
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        print("BPE tokenizer files not found")
        return None
    
    if (os.path.getsize(vocab_path) == 0 or os.path.getsize(merges_path) == 0):
        print("BPE tokenizer files are empty or corrupted")
        return None
    
    try:
        # Load from vocab.json and merges.txt for consistency
        bpe_tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)
        
        # Set post-processing template
        bpe_tokenizer.post_processor = TemplateProcessing(
            single=f"{SOS_TOKEN} $A {EOS_TOKEN}",
            pair=f"{SOS_TOKEN} $A {EOS_TOKEN} $B {EOS_TOKEN}",
            special_tokens=[(SOS_TOKEN, bpe_tokenizer.token_to_id(SOS_TOKEN)), 
                           (EOS_TOKEN, bpe_tokenizer.token_to_id(EOS_TOKEN))],
        )
        return bpe_tokenizer
    except Exception as e:
        print(f"Error loading BPE tokenizer: {e}")
        return None

def text_to_bpe_tokens(text, add_special_tokens=True):
    """Convert text to BPE token IDs"""
    global bpe_tokenizer
    
    if bpe_tokenizer is None:
        bpe_tokenizer = load_bpe_tokenizer()
    
    if bpe_tokenizer is None:
        # Fallback to word-level tokenization
        words = clean_text(text).split()
        if add_special_tokens:
            words = [SOS_TOKEN] + words + [EOS_TOKEN]
        # Create a simple mapping for special tokens
        special_token_ids = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
        return [special_token_ids.get(w, special_token_ids[UNK_TOKEN]) for w in words]
    
    encoding = bpe_tokenizer.encode(text)
    if add_special_tokens:
        # Manually add SOS and EOS tokens
        sos_id = bpe_tokenizer.token_to_id(SOS_TOKEN)
        eos_id = bpe_tokenizer.token_to_id(EOS_TOKEN)
        if sos_id is None or eos_id is None:
            # Fallback if special tokens not in tokenizer
            return [0] + encoding.ids + [1]
        return [sos_id] + encoding.ids + [eos_id]
    return encoding.ids

def bpe_tokens_to_text(token_ids):
    """Convert BPE token IDs back to text"""
    global bpe_tokenizer
    
    if bpe_tokenizer is None:
        bpe_tokenizer = load_bpe_tokenizer()
    
    if bpe_tokenizer is None:
        # Fallback to word-level tokenization
        # Create a simple mapping for special tokens
        special_tokens = {i: token for i, token in enumerate(SPECIAL_TOKENS)}
        tokens = [special_tokens.get(idx, UNK_TOKEN) for idx in token_ids]
        return " ".join([t for t in tokens if t not in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]])
    
    # Filter out special tokens before decoding
    sos_id = bpe_tokenizer.token_to_id(SOS_TOKEN)
    eos_id = bpe_tokenizer.token_to_id(EOS_TOKEN)
    pad_id = bpe_tokenizer.token_to_id(PAD_TOKEN)
    
    filtered_ids = [id for id in token_ids if id not in [sos_id, eos_id, pad_id] and id is not None]
    return bpe_tokenizer.decode(filtered_ids, skip_special_tokens=True)

# ------------------ Text preprocessing ------------------
def clean_text(text):
    """Lowercase and remove special characters, keeping basic punctuation"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Keep letters, numbers, basic punctuation, and special tokens
    text = re.sub(r"[^a-z0-9<>\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ------------------ Data Augmentation ------------------
def synonym_replacement(text, n=1):
    """Replace n words in the text with their synonyms"""
    words = word_tokenize(text)
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in SPECIAL_TOKENS]))
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    return " ".join(new_words)

def get_synonyms(word):
    """Get synonyms for a word using WordNet"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            if synonym != word and len(synonym.split()) == 1:
                synonyms.add(synonym)
    return synonyms

def augment_text(text, augmentation_prob=0.3):
    """Apply various text augmentation techniques"""
    if random.random() > augmentation_prob:
        return text
    
    augmentation_type = random.choice(["synonym", "dropout", "swap"])
    
    if augmentation_type == "synonym" and len(text.split()) > 2:
        # Replace up to 20% of words with synonyms
        n = max(1, int(len(text.split()) * 0.2))
        return synonym_replacement(text, n)
    
    elif augmentation_type == "dropout":
        # Randomly drop some words
        words = text.split()
        if len(words) > 3:
            # Drop 10-30% of words
            n = random.randint(max(1, int(len(words) * 0.1)), max(1, int(len(words) * 0.3)))
            indices_to_drop = random.sample(range(len(words)), n)
            words = [word for i, word in enumerate(words) if i not in indices_to_drop]
            return " ".join(words)
    
    elif augmentation_type == "swap" and len(text.split()) > 3:
        # Randomly swap adjacent words
        words = text.split()
        swap_idx = random.randint(0, len(words) - 2)
        words[swap_idx], words[swap_idx + 1] = words[swap_idx + 1], words[swap_idx]
        return " ".join(words)
    
    return text

# ------------------ Context Building ------------------
def build_context_from_memory():
    """Build context string from session memory with personality token"""
    if not session_memory:
        return CURRENT_PERSONALITY
    
    # Get the last N exchanges
    recent_exchanges = session_memory[-MAX_SESSION_TURNS:]
    context_parts = [CURRENT_PERSONALITY]
    
    for user_input, response in recent_exchanges:
        # Add intent token based on user input type
        intent = detect_intent(user_input)
        context_parts.append(f"{intent} user: {user_input}")
        context_parts.append(f"assistant: {response}")
    
    return " ".join(context_parts)

def detect_intent(text):
    """Detect the intent of the user input"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ["what", "why", "how", "when", "where", "who", "which", "?"]):
        return "<QUESTION>"
    elif any(word in text_lower for word in ["hi", "hello", "hey", "greetings"]):
        return "<GREETING>"
    elif any(word in text_lower for word in ["bye", "goodbye", "see you", "farewell"]):
        return "<FAREWELL>"
    elif any(word in text_lower for word in ["please", "could you", "would you", "do this"]):
        return "<COMMAND>"
    else:
        return "<STATEMENT>"

# ------------------ Thinking/Debug Print ------------------
def print_thinking(context, response, generated_tokens):
    """Print debug information about the thinking process"""
    if not DEBUG_THINKING:
        return
    
    print("\n🤔 LYNQBIT THINKING:")
    print(f"🎭 Current personality: {CURRENT_PERSONALITY} {PERSONALITY_TOKENS.get(CURRENT_PERSONALITY, '')}")
    print(f"📝 Context: {context}")
    print(f"🧠 Token-by-token generation:")
    for i, token_id in enumerate(generated_tokens):
        token_text = bpe_tokens_to_text([token_id]) if USE_BPE_TOKENIZATION else idx2word.get(token_id, UNK_TOKEN)
        print(f"  Step {i+1}: {token_id} -> '{token_text}'")
    print(f"💬 Final response: {response}")
    print("---\n")

# ------------------ Online Learning ------------------
def online_learning_update(user_input, response):
    """Update the model with the latest exchange (online learning)"""
    if not MINI_REPLAY_ENABLED:
        return
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
    
    if USE_BPE_TOKENIZATION and bpe_tokenizer:
        criterion = nn.CrossEntropyLoss(ignore_index=bpe_tokenizer.token_to_id(PAD_TOKEN))
        src_tensor = torch.tensor([text_to_bpe_tokens(user_input, add_special_tokens=True)], device=device)
        tgt_tensor = torch.tensor([text_to_bpe_tokens(response, add_special_tokens=True)], device=device)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=word2idx[PAD_TOKEN])
        src_text = clean_text(user_input)
        tgt_text = clean_text(response)
        src_tensor = text_to_tensor([SOS_TOKEN] + src_text.split() + [EOS_TOKEN], word2idx).unsqueeze(0)
        tgt_tensor = text_to_tensor([SOS_TOKEN] + tgt_text.split() + [EOS_TOKEN], word2idx).unsqueeze(0)
    
    # Forward pass
    if USE_BPE_TOKENIZATION and bpe_tokenizer:
        src_padding_mask = (src_tensor == bpe_tokenizer.token_to_id(PAD_TOKEN)).to(device)
        tgt_padding_mask = (tgt_tensor[:, :-1] == bpe_tokenizer.token_to_id(PAD_TOKEN)).to(device)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tensor[:, :-1].size(1)).to(device)
    else:
        src_padding_mask, tgt_padding_mask, tgt_mask = create_mask(src_tensor, tgt_tensor[:, :-1], word2idx)
    
    output = model(
        src_tensor,
        tgt_tensor[:, :-1],
        src_key_padding_mask=src_padding_mask,
        tgt_key_padding_mask=tgt_padding_mask,
        tgt_mask=tgt_mask,
    )
    
    # Calculate loss
    if USE_BPE_TOKENIZATION and bpe_tokenizer:
        loss = criterion(output.reshape(-1, bpe_tokenizer.get_vocab_size()), tgt_tensor[:, 1:].reshape(-1))
    else:
        loss = criterion(output.reshape(-1, vocab_size), tgt_tensor[:, 1:].reshape(-1))
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()
    
    model.eval()
    print(f"📚 Online learning update applied (loss: {loss.item():.4f})")

# ------------------ Logging ------------------
def log_to_csv(user_input, response):
    """Log conversation to CSV file with metadata"""
    if not LOGGING_ENABLED:
        return
    
    file_exists = os.path.isfile('conversation_log.csv')
    
    with open('conversation_log.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'user_input', 'model_response', 'personality', 'intent', 'session_length'])
        
        intent = detect_intent(user_input)
        writer.writerow([
            datetime.now().isoformat(),
            user_input,
            response,
            CURRENT_PERSONALITY,
            intent,
            len(session_memory)
        ])

# ------------------ Helpers ------------------
def text_to_tensor(words, word2idx):
    """words: list of tokens (strings)"""
    idxs = [word2idx.get(w, word2idx[UNK_TOKEN]) for w in words]
    return torch.tensor(idxs, dtype=torch.long, device=device)

def tensor_to_text(tensor, idx2word):
    words = []
    for i in tensor:
        w = idx2word.get(int(i.item()), UNK_TOKEN)
        if w not in (SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN):
            words.append(w)
    return " ".join(words)

def create_mask(src, tgt, word2idx):
    tgt_seq_len = tgt.shape[1]
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)

    src_padding_mask = (src == word2idx[PAD_TOKEN]).to(device)
    tgt_padding_mask = (tgt == word2idx[PAD_TOKEN]).to(device)

    return src_padding_mask, tgt_padding_mask, tgt_mask

def create_bpe_mask(src, tgt, tokenizer):
    tgt_seq_len = tgt.shape[1]
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)

    src_padding_mask = (src == tokenizer.token_to_id(PAD_TOKEN)).to(device)
    tgt_padding_mask = (tgt == tokenizer.token_to_id(PAD_TOKEN)).to(device)

    return src_padding_mask, tgt_padding_mask, tgt_mask

# ------------------ Word Dropout ------------------
def apply_word_dropout(batch, word2idx, rate=0.0):
    if rate <= 0.0:
        return batch
    mask = (torch.rand(batch.shape, device=batch.device) < rate) & (batch != word2idx[PAD_TOKEN])
    dropped = batch.clone()
    dropped[mask] = word2idx[UNK_TOKEN]
    return dropped

def apply_bpe_dropout(batch, tokenizer, rate=0.0):
    if rate <= 0.0:
        return batch
    mask = (torch.rand(batch.shape, device=batch.device) < rate) & (batch != tokenizer.token_to_id(PAD_TOKEN))
    dropped = batch.clone()
    dropped[mask] = tokenizer.token_to_id(UNK_TOKEN)
    return dropped

# ------------------ Nucleus sampling util ------------------
def top_p_filter(probs, top_p):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs > top_p
    mask[..., 0] = False
    indices_to_remove = torch.zeros_like(mask, dtype=torch.bool).scatter(1, sorted_indices, mask)
    filtered = probs.masked_fill(indices_to_remove, 0.0)
    if filtered.sum() == 0:
        return probs
    return filtered / filtered.sum()

# ------------------ Generation Helper ------------------
def generate_response(src, model, tokenizer, max_length=256, temperature_local=1.0, top_p_local=0.9):
    """Generate response from source tokens"""
    model.eval()
    
    with torch.no_grad():
        # Start with SOS token
        tgt = torch.tensor([[tokenizer.token_to_id(SOS_TOKEN)]], device=device)
        
        generated_tokens = []
        
        for step in range(max_length):
            # Create masks
            src_padding_mask = (src == tokenizer.token_to_id(PAD_TOKEN)).to(device)
            tgt_padding_mask = (tgt == tokenizer.token_to_id(PAD_TOKEN)).to(device)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)
            
            # Get output from model
            output = model(
                src,
                tgt,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                tgt_mask=tgt_mask,
            )
            
            next_token_logits = output[:, -1, :] / max(1e-8, temperature_local)
            probs = torch.softmax(next_token_logits, dim=-1)
            probs = top_p_filter(probs, top_p_local)
            
            next_token = torch.multinomial(probs, 1)
            next_token_id = int(next_token.item())
            
            # Check for EOS token
            if next_token_id == tokenizer.token_to_id(EOS_TOKEN):
                break
                
            generated_tokens.append(next_token_id)
            tgt = torch.cat([tgt, next_token], dim=1)
        
        # Convert tokens to text
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response

# ------------------ Generation with Thinking Logs ------------------
def generate_coherent_response(
    input_text,
    model,
    word2idx=None,
    idx2word=None,
    max_length=256,
    temperature_local=temperature,
    top_p_local=top_p,
    repetition_penalty_local=repetition_penalty,
):
    model.eval()
    
    if USE_BPE_TOKENIZATION and bpe_tokenizer:
        tokens = text_to_bpe_tokens(input_text, add_special_tokens=False)
        tokens = tokens[-(max_len - 2):]
        src = torch.tensor([tokens + [bpe_tokenizer.token_to_id(EOS_TOKEN)]], device=device)
        vocab_size = bpe_tokenizer.get_vocab_size()
    else:
        cleaned = clean_text(input_text)
        tokens = cleaned.split() if cleaned else []
        tokens = tokens[-(max_len - 2):]
        src = text_to_tensor([SOS_TOKEN] + tokens + [EOS_TOKEN], word2idx).unsqueeze(0)
        vocab_size = len(word2idx)
    
    generated_tokens = []
    last_tokens = []
    thinking_log = []

    with torch.no_grad():
        # Start with SOS token
        if USE_BPE_TOKENIZATION and bpe_tokenizer:
            tgt = torch.tensor([[bpe_tokenizer.token_to_id(SOS_TOKEN)]], device=device)
        else:
            tgt = torch.tensor([[word2idx[SOS_TOKEN]]], device=device)
            
        for step in range(max_length):
            if USE_BPE_TOKENIZATION and bpe_tokenizer:
                src_padding_mask, tgt_padding_mask, tgt_mask = create_bpe_mask(src, tgt, bpe_tokenizer)
            else:
                src_padding_mask, tgt_padding_mask, tgt_mask = create_mask(src, tgt, word2idx)
                
            # Get output from model
            output = model(
                src,
                tgt,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                tgt_mask=tgt_mask,
            )
            
            next_token_logits = output[:, -1, :] / max(1e-8, temperature_local)

            # Apply repetition penalty
            if USE_BPE_TOKENIZATION and bpe_tokenizer:
                sos_token_id = bpe_tokenizer.token_to_id(SOS_TOKEN)
            else:
                sos_token_id = word2idx[SOS_TOKEN]
                
            for token_id in set(tgt[0].tolist() + last_tokens[-5:]):
                if token_id != sos_token_id:
                    next_token_logits[0, token_id] = next_token_logits[0, token_id] / (repetition_penalty_local * 1.5)

            probs = torch.softmax(next_token_logits, dim=-1)
            probs = top_p_filter(probs, top_p_local)

            if temperature_local < 1e-4:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                if probs.sum() <= 0:
                    probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

            next_token_id = int(next_token.item())
            generated_tokens.append(next_token_id)
            
            # Log the thinking process
            if USE_BPE_TOKENIZATION and bpe_tokenizer:
                next_word = bpe_tokens_to_text([next_token_id])
                eos_token_id = bpe_tokenizer.token_to_id(EOS_TOKEN)
                pad_token_id = bpe_tokenizer.token_to_id(PAD_TOKEN)
                unk_token_id = bpe_tokenizer.token_to_id(UNK_TOKEN)
            else:
                next_word = idx2word.get(next_token_id, UNK_TOKEN)
                eos_token_id = word2idx[EOS_TOKEN]
                pad_token_id = word2idx[PAD_TOKEN]
                unk_token_id = word2idx[UNK_TOKEN]
                
            thinking_log.append(f"Step {step+1}: Token {next_token_id} -> '{next_word}'")
            
            if next_token_id in (eos_token_id, pad_token_id, unk_token_id):
                break

            tgt = torch.cat([tgt, next_token], dim=1)
            last_tokens.append(next_token_id)

            if next_token_id == eos_token_id:
                break

    if USE_BPE_TOKENIZATION and bpe_tokenizer:
        response = bpe_tokens_to_text(generated_tokens)
    else:
        response = tensor_to_text(torch.tensor(generated_tokens), idx2word)
    
    # Print thinking process
    if DEBUG_THINKING:
        print_thinking(input_text, response, generated_tokens)
    
    return response

# ------------------ Web Scraping for Data Collection ------------------
def scrape_conversation_data(url, max_pages=5):
    """Scrape conversation data from websites for training"""
    conversations = []
    
    try:
        for page in range(max_pages):
            response = requests.get(f"{url}?page={page+1}")
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find conversation elements (this will vary by website)
            conversation_elements = soup.find_all('div', class_='conversation')
            
            for elem in conversation_elements:
                user_msg = elem.find('div', class_='user-message')
                assistant_msg = elem.find('div', class_='assistant-message')
                
                if user_msg and assistant_msg:
                    conversations.append({
                        "user": clean_text(user_msg.get_text()),
                        "assistant": clean_text(assistant_msg.get_text())
                    })
    except Exception as e:
        print(f"Web scraping failed: {e}")
    
    return conversations

# ------------------ Training / Inference ------------------
if TRAIN_MODE:
    # First pretrain on English corpus if enabled
    pretrained_model = None
    if ENGLISH_MODEL_TRAIN and os.path.exists("english_model.txt"):
        pretrained_model, bpe_tokenizer = pretrain_model("english_model.txt", epochs=10)
        print("✅ Pretraining completed on English corpus")
    else:
        print("⚠️  English model training skipped or english_model.txt not found")

    # Load training data
    try:
        from pymongo import MongoClient
        from dotenv import load_dotenv

        load_dotenv()
        MONGO_URI = os.getenv("MONGO_URI")
        if not MONGO_URI:
            raise RuntimeError("MONGO_URI not set")
        client = MongoClient(MONGO_URI)
        db = client["lynqbit_db"]
        collection = db["training_data"]
        raw_data = list(collection.find())
        if not raw_data:
            raise RuntimeError("No data found in MongoDB collection 'training_data'")
        
        # Convert to multi-turn format
        conversations = []
        current_conversation = []
        
        for i, d in enumerate(raw_data):
            question = clean_text(d.get("question", ""))
            answer = clean_text(d.get("answer", ""))
            
            current_conversation.append({"user": question, "assistant": answer})
            
            # Start a new conversation every 3-5 exchanges
            if len(current_conversation) >= random.randint(3, 5) or i == len(raw_data) - 1:
                conversations.append(current_conversation)
                current_conversation = []
                
    except Exception as e:
        print(f"[warning] MongoDB not available or failed: {e}. Trying local fallback...")
        
        # Try to load multi-turn conversations from a JSON file
        if os.path.exists("multi_turn_conversations.json"):
            with open("multi_turn_conversations.json", "r", encoding="utf-8") as f:
                conversations = json.load(f)
        else:
            # Fallback to single-turn format
            conversations = []
            if os.path.exists("training_data.txt"):
                with open("training_data.txt", "r", encoding="utf-8") as f:
                    current_conversation = []
                    for line in f:
                        if "\t" in line:
                            q, a = line.strip().split("\t", 1)
                        else:
                            parts = line.strip().split("||")
                            if len(parts) >= 2:
                                q, a = parts[0], parts[1]
                            else:
                                continue
                        
                        current_conversation.append({"user": clean_text(q), "assistant": clean_text(a)})
                        
                        # Start a new conversation every 3-5 exchanges
                        if len(current_conversation) >= random.randint(3, 5):
                            conversations.append(current_conversation)
                            current_conversation = []
                    
                    if current_conversation:
                        conversations.append(current_conversation)
            else:
                raise RuntimeError("No training data found")

    print(f"Loaded {len(conversations)} multi-turn conversations")

    # Prepare texts for BPE tokenizer training
    all_texts = []
    for conv in conversations:
        for exchange in conv:
            all_texts.append(exchange["user"])
            all_texts.append(exchange["assistant"])
    
    # Add personality tokens to training data
    for personality in PERSONALITY_TOKENS:
        all_texts.append(personality)
    
    # Add intent tokens to training data
    for intent in INTENT_TOKENS:
        all_texts.append(intent)

    # Train or load BPE tokenizer
# ... (previous code continues)

    # Train or load BPE tokenizer
    if USE_BPE_TOKENIZATION:
        # If we have a pretrained tokenizer, use it
        if bpe_tokenizer is not None:
            print("Using pretrained BPE tokenizer")
        else:
            # Check if tokenizer files exist and are valid
            vocab_path = "tokenizer/vocab.json"
            merges_path = "tokenizer/merges.txt"
            
            tokenizer_valid = (
                os.path.exists(vocab_path) and 
                os.path.exists(merges_path) and 
                os.path.getsize(vocab_path) > 0 and 
                os.path.getsize(merges_path) > 0
            )
            
            if tokenizer_valid:
                bpe_tokenizer = load_bpe_tokenizer()
                if bpe_tokenizer is not None:
                    print("Loaded pre-trained BPE tokenizer")
                else:
                    print("Failed to load BPE tokenizer, training a new one...")
                    bpe_tokenizer = train_bpe_tokenizer(all_texts)
            else:
                print("Training new BPE tokenizer...")
                bpe_tokenizer = train_bpe_tokenizer(all_texts)
                print("Trained new BPE tokenizer")
        
        if bpe_tokenizer is not None:
            vocab_size = bpe_tokenizer.get_vocab_size()
            print(f"BPE Vocabulary size: {vocab_size}")
            
            # Verify tokenizer works
            print("\nVerifying tokenizer...")
            sample_text = "What is your name?"
            tokens = text_to_bpe_tokens(sample_text)
            decoded = bpe_tokens_to_text(tokens)
            print(f"Sample: '{sample_text}'")
            print(f"Tokens: {tokens}")
            print(f"Decoded: '{decoded}'")
        else:
            print("BPE tokenization failed, falling back to word-level tokenization")
            USE_BPE_TOKENIZATION = False
    
    if not USE_BPE_TOKENIZATION:
        # Word-level tokenization (original approach)
        counter = Counter()
        for text in all_texts:
            counter.update(text.split())
        
        min_freq = 2
        kept_tokens = [w for w, c in counter.items() if c >= min_freq]
        special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + list(PERSONALITY_TOKENS.keys()) + list(INTENT_TOKENS.keys())
        all_words = special_tokens + sorted([t for t in kept_tokens if t not in special_tokens])
        
        word2idx = {w: i for i, w in enumerate(all_words)}
        idx2word = {i: w for w, i in word2idx.items()}
        vocab_size = len(word2idx)
        print(f"Word-level Vocabulary size: {vocab_size}")

    # Create multi-turn dataset
    dataset = MultiTurnDataset(conversations, bpe_tokenizer if USE_BPE_TOKENIZATION else word2idx)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    def collate_fn(batch):
        contexts, targets = zip(*batch)
        
        if USE_BPE_TOKENIZATION and bpe_tokenizer:
            # Tokenize with BPE
            src_tensors = [torch.tensor(text_to_bpe_tokens(ctx, add_special_tokens=True), device=device) for ctx in contexts]
            tgt_tensors = [torch.tensor(text_to_bpe_tokens(tgt, add_special_tokens=True), device=device) for tgt in targets]
            padding_value = bpe_tokenizer.token_to_id(PAD_TOKEN)
        else:
            # Tokenize with word-level tokenization
            src_tensors = [text_to_tensor([SOS_TOKEN] + ctx.split() + [EOS_TOKEN], word2idx) for ctx in contexts]
            tgt_tensors = [text_to_tensor([SOS_TOKEN] + tgt.split() + [EOS_TOKEN], word2idx) for tgt in targets]
            padding_value = word2idx[PAD_TOKEN]
            
        # Pad sequences
        src_batch = pad_sequence(src_tensors, batch_first=True, padding_value=padding_value)
        tgt_batch = pad_sequence(tgt_tensors, batch_first=True, padding_value=padding_value)
        
        # Truncate to max_len
        src_batch = src_batch[:, :max_len]
        tgt_batch = tgt_batch[:, :max_len]
        
        return src_batch, tgt_batch

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    padding_idx = bpe_tokenizer.token_to_id(PAD_TOKEN) if (USE_BPE_TOKENIZATION and bpe_tokenizer) else word2idx[PAD_TOKEN]
    
    # Use pretrained model if available, otherwise create a new one
    if pretrained_model is not None:
        model = pretrained_model
        print("Using pretrained model for fine-tuning")
    else:
        model = ImprovedTransformer(vocab_size, d_model, num_heads, num_layers, max_len, 
                                   dropout=dropout_rate, 
                                   padding_idx=padding_idx).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.8)
    
    epochs = 200
    min_val_loss = float("inf")
    patience_counter = 0

    print("Starting training with improved transformer...")

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0

        for src_batch, tgt_batch in train_loader:
            # Apply data augmentation with 30% probability
            if random.random() < 0.3:
                augmented_src = []
                for i in range(src_batch.size(0)):
                    if USE_BPE_TOKENIZATION and bpe_tokenizer:
                        text = bpe_tokens_to_text(src_batch[i].tolist())
                    else:
                        text = tensor_to_text(src_batch[i], idx2word)
                    augmented_text = augment_text(text)
                    if USE_BPE_TOKENIZATION and bpe_tokenizer:
                        augmented_tokens = text_to_bpe_tokens(augmented_text, add_special_tokens=True)
                        augmented_src.append(torch.tensor(augmented_tokens, device=device))
                    else:
                        augmented_src.append(text_to_tensor([SOS_TOKEN] + augmented_text.split() + [EOS_TOKEN], word2idx))
                
                src_batch = pad_sequence(augmented_src, batch_first=True, padding_value=padding_idx)
                src_batch = src_batch[:, :max_len]

            # Apply word dropout
            if USE_BPE_TOKENIZATION and bpe_tokenizer:
                src_batch_aug = apply_bpe_dropout(src_batch, bpe_tokenizer, rate=word_dropout_rate)
                tgt_in = apply_bpe_dropout(tgt_batch[:, :-1], bpe_tokenizer, rate=word_dropout_rate)
            else:
                src_batch_aug = apply_word_dropout(src_batch, word2idx, rate=word_dropout_rate)
                tgt_in = apply_word_dropout(tgt_batch[:, :-1], word2idx, rate=word_dropout_rate)

            # Create masks
            if USE_BPE_TOKENIZATION and bpe_tokenizer:
                src_padding_mask, tgt_padding_mask, tgt_mask = create_bpe_mask(src_batch_aug, tgt_in, bpe_tokenizer)
            else:
                src_padding_mask, tgt_padding_mask, tgt_mask = create_mask(src_batch_aug, tgt_in, word2idx)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(
                src_batch_aug,
                tgt_in,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                tgt_mask=tgt_mask,
            )

            # Calculate loss
            loss = criterion(output.reshape(-1, vocab_size), tgt_batch[:, 1:].reshape(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_train_loss += float(loss.item())
            num_train_batches += 1

        avg_train_loss = total_train_loss / (num_train_batches if num_train_batches > 0 else 1)
        
        # Validation phase
        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for src_batch, tgt_batch in val_loader:
                if USE_BPE_TOKENIZATION and bpe_tokenizer:
                    src_padding_mask, tgt_padding_mask, tgt_mask = create_bpe_mask(src_batch, tgt_batch[:, :-1], bpe_tokenizer)
                else:
                    src_padding_mask, tgt_padding_mask, tgt_mask = create_mask(src_batch, tgt_batch[:, :-1], word2idx)
                
                output = model(
                    src_batch,
                    tgt_batch[:, :-1],
                    src_key_padding_mask=src_padding_mask,
                    tgt_key_padding_mask=tgt_padding_mask,
                    tgt_mask=tgt_mask,
                )
                
                loss = criterion(output.reshape(-1, vocab_size), tgt_batch[:, 1:].reshape(-1))
                total_val_loss += float(loss.item())
                num_val_batches += 1
        
        avg_val_loss = total_val_loss / (num_val_batches if num_val_batches > 0 else 1)
        scheduler.step(avg_val_loss)

        # Checkpointing based on validation loss
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            patience_counter = 0
            # Save model with all necessary components
            save_data = {
                "model_state": model.state_dict(),
                "d_model": d_model,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "max_len": max_len,
                "use_bpe": USE_BPE_TOKENIZATION
            }
            
            if USE_BPE_TOKENIZATION and bpe_tokenizer:
                # For BPE, we need to save the tokenizer info
                save_data["tokenizer_path"] = "tokenizer"
            else:
                # For word-level, save the vocabulary
                save_data["word2idx"] = word2idx
                save_data["idx2word"] = idx2word
            
            torch.save(save_data, "best_model.pth")
        else:
            patience_counter += 1

        # Print progress
        if (epoch + 1) % 1 == 0:
            print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Test generation with different personalities
            test_prompts = ["hello", "who created you", "tell me a story"]
            personalities = list(PERSONALITY_TOKENS.keys())[:3]  # Test with first 3 personalities
            
            for personality in personalities:
                print(f"\n🎭 Testing with {personality}:")
                for prompt in test_prompts:
                    context = f"{personality} user: {clean_text(prompt)}"
                    reply = generate_coherent_response(
                        context, 
                        model, 
                        word2idx if not USE_BPE_TOKENIZATION else None,
                        idx2word if not USE_BPE_TOKENIZATION else None,
                        max_length=256, 
                        temperature_local=1.0
                    )
                    print(f"  '{prompt}' -> '{reply}'")
            print()
            
            # Switch back to training mode
            model.train()

    # Load best model after training
    if os.path.exists("best_model.pth"):
        checkpoint = torch.load("best_model.pth", map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        USE_BPE_TOKENIZATION = checkpoint.get("use_bpe", USE_BPE_TOKENIZATION)
        
        if not USE_BPE_TOKENIZATION:
            word2idx = checkpoint["word2idx"]
            idx2word = checkpoint["idx2word"]
        
        model.eval()
        print("Best model loaded into memory (ready for inference).")

else:

    # Ensure all globals are initialized before use
    if not os.path.exists("best_model.pth"):
        raise FileNotFoundError("best_model.pth not found. Train first or provide checkpoint.")
    checkpoint = torch.load("best_model.pth", map_location=device)
    USE_BPE_TOKENIZATION = checkpoint.get("use_bpe", USE_BPE_TOKENIZATION)

    if USE_BPE_TOKENIZATION:
        bpe_tokenizer = load_bpe_tokenizer()
        if bpe_tokenizer is not None:
            vocab_size = bpe_tokenizer.get_vocab_size()
        else:
            print("BPE tokenizer failed to load, falling back to word-level tokenization")
            USE_BPE_TOKENIZATION = False

    if not USE_BPE_TOKENIZATION:
        if "word2idx" in checkpoint and "idx2word" in checkpoint:
            word2idx = checkpoint["word2idx"]
            idx2word = checkpoint["idx2word"]
            vocab_size = len(word2idx)
        else:
            print("Word-level vocabulary not found in checkpoint, creating a simple one")
            special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + list(PERSONALITY_TOKENS.keys()) + list(INTENT_TOKENS.keys())
            word2idx = {w: i for i, w in enumerate(special_tokens)}
            idx2word = {i: w for w, i in word2idx.items()}
            vocab_size = len(word2idx)

    padding_idx = bpe_tokenizer.token_to_id(PAD_TOKEN) if (USE_BPE_TOKENIZATION and bpe_tokenizer) else word2idx[PAD_TOKEN]
    model = ImprovedTransformer(
        vocab_size,
        checkpoint["d_model"],
        checkpoint["num_heads"],
        checkpoint["num_layers"],
        checkpoint["max_len"],
        dropout=dropout_rate,
        padding_idx=padding_idx
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print("Loaded model from best_model.pth")

    import gradio as gr

    # ------------------ Session Memory ------------------
    session_memory = []
    MAX_SESSION_TURNS = 50

    # ------------------ Chat Helpers ------------------
    def clean_text(text):
        return text.strip()

    # ------------------ Chat Function ------------------
    def chat(user_input, history):
        global session_memory, CURRENT_PERSONALITY

        # Handle clear command
        if user_input.strip().lower() == "!clear":
            session_memory = []
            return [], []

        # Handle personality switch
        if user_input.strip().lower().startswith("!personality"):
            parts = user_input.split()
            if len(parts) > 1:
                new_personality = f"<{parts[1].upper()}>"
                if new_personality in PERSONALITY_TOKENS:
                    CURRENT_PERSONALITY = new_personality
                    response = f"Personality changed to {new_personality} {PERSONALITY_TOKENS[new_personality]}"
                    history = history or []
                    history.append((user_input, response))
                    return history, history
                else:
                    response = f"Unknown personality. Available: {', '.join(PERSONALITY_TOKENS.keys())}"
                    history = history or []
                    history.append((user_input, response))
                    return history, history

        cleaned_input = clean_text(user_input)

        # Build context from session memory
        context = build_context_from_memory()
        full_context = f"{context} {detect_intent(cleaned_input)} user: {cleaned_input}"

        # Generate response using the main model
        response = generate_coherent_response(
            full_context,
            model,
            word2idx if not USE_BPE_TOKENIZATION else None,
            idx2word if not USE_BPE_TOKENIZATION else None,
            max_length=256,
            temperature_local=temperature,
            top_p_local=top_p,
            repetition_penalty_local=repetition_penalty
        )

        # Add personality emoji
        personality_emoji = PERSONALITY_TOKENS.get(CURRENT_PERSONALITY, "")
        if response and personality_emoji:
            response = f"{response} {personality_emoji}"

        # Update session memory
        session_memory.append((cleaned_input, response))

        # Mini replay / online learning
        if MINI_REPLAY_ENABLED:
            online_learning_update(cleaned_input, response)

        # Logging
        if LOGGING_ENABLED:
            log_to_csv(user_input, response)

        # Update chat history
        history = history or []
        history.append((user_input, response))
        return history, history

    # ------------------ Gradio UI ------------------
    with gr.Blocks() as demo:
        gr.Markdown("## 😼 Lynqbit Chatbot with Enhanced Features")
        gr.Markdown("**Features:** BPE Tokenization | Multi-turn Conversations | Personality Tokens | Data Augmentation")

        # Personality selector
        gr.Markdown("### 🎭 Select Personality")
        personality_radio = gr.Radio(
            choices=list(PERSONALITY_TOKENS.keys()),
            value=CURRENT_PERSONALITY,
            label="Personality"
        )

        def update_personality(personality):
            global CURRENT_PERSONALITY
            CURRENT_PERSONALITY = personality
            return f"Personality set to {personality} {PERSONALITY_TOKENS[personality]}"

        personality_radio.change(update_personality, inputs=personality_radio, outputs=gr.Textbox(visible=False))

        # Chat interface
        chatbot = gr.Chatbot(label="Conversation History", height=500)
        msg = gr.Textbox(label="Your Message", placeholder="Type your message here...")
        clear_btn = gr.Button("Clear Chat")

        def respond(message, chat_history):
            return chat(message, chat_history)

        msg.submit(respond, [msg, chatbot], [chatbot, msg])
        clear_btn.click(lambda: ([], []), None, [chatbot, msg], queue=False)

        demo.launch(server_name="0.0.0.0", server_port=7860)
