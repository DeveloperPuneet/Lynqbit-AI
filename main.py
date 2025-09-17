# main.py
import torch  # Import PyTorch 🤖
import torch.nn as nn  # Neural network module 🧠
import torch.optim as optim  # For optimization ⚙️
from torch.nn.utils.rnn import pad_sequence  # Pad sequences 📏
import random  # For randomness 🎲
import numpy as np  # For numerical ops 🧮
import os  # For OS interaction 💻
import re  # For text cleaning 🧹
from collections import Counter

# ------------------ Ephemeral session memory ------------------
session_memory = []  # stores (user_input, lynqbit_response)
MAX_SESSION_TURNS = 50  # number of recent turns that influence responses


# ------------------ Flags ------------------
TRAIN_MODE = False  # set to False if you only want to chat with the saved model

# ------------------ Special tokens ------------------
SOS_TOKEN = "<SOS>"  # Start token 🏁
EOS_TOKEN = "<EOS>"  # End token 🏁
PAD_TOKEN = "<PAD>"  # Padding token 🧱
UNK_TOKEN = "<UNK>"  # Unknown token ❓

# ------------------ Model parameters ------------------
d_model = 256  # Embedding dimension 📏
num_heads = 8  # Attention heads count 🧠
num_layers = 4  # Transformer layers count 🧱
max_len = 60  # Maximum sequence length 📏 (tokens)
batch_size = 16  # Batch size 📦
temperature = 1.0  # Sampling temperature 🌡️
top_p = 0.92  # Nucleus sampling parameter ☢️
repetition_penalty = 1.2  # Repetition penalty factor ⚖️
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available 🚀
dropout_rate = 0.3  # Increased dropout for better generalization 💧
word_dropout_rate = 0.1  # randomly replace words with <UNK> during training

# ------------------ Text preprocessing ------------------
def clean_text(text):
    """Lowercase and remove most special characters"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9<>\s.!?]", " ", text)  # allow angle tokens if any, keep punctuation .,!? and alphanum
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------ Conversation History ------------------
conversation_history = []  # store conversation for chat mode

# ------------------ Model ------------------
class ImprovedTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads=8, num_layers=4, max_len=60, dropout=0.1, padding_idx=None):
        super().__init__()
        # note: padding_idx cannot be set later easily; if you want embedding padding, pass padding_idx here
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
        # src: (B, S), tgt: (B, T)
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
    """
    Return:
      src_key_padding_mask: (B, S) bool [True for positions that should be masked/ignored]
      tgt_key_padding_mask: (B, T) bool
      tgt_mask: (T, T) float causal mask (upper triangle filled with -inf)
    Note: PyTorch's Transformer expects boolean padding masks.
    """
    tgt_seq_len = tgt.shape[1]
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)

    src_padding_mask = (src == word2idx[PAD_TOKEN]).to(device)  # bool
    tgt_padding_mask = (tgt == word2idx[PAD_TOKEN]).to(device)  # bool

    return src_padding_mask, tgt_padding_mask, tgt_mask


# ------------------ Word Dropout (training-time augmentation) ------------------
def apply_word_dropout(batch, word2idx, rate=0.0):
    """
    batch: LongTensor (B, S)
    replace tokens randomly with UNK (except PAD)
    """
    if rate <= 0.0:
        return batch
    mask = (torch.rand(batch.shape, device=batch.device) < rate) & (batch != word2idx[PAD_TOKEN])
    dropped = batch.clone()
    dropped[mask] = word2idx[UNK_TOKEN]
    return dropped


# ------------------ Nucleus sampling util ------------------
def top_p_filter(probs, top_p):
    """
    probs: (1, V)
    """
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs > top_p
    mask[..., 0] = False  # always keep top
    indices_to_remove = torch.zeros_like(mask, dtype=torch.bool).scatter(1, sorted_indices, mask)
    filtered = probs.masked_fill(indices_to_remove, 0.0)
    if filtered.sum() == 0:
        return probs
    return filtered / filtered.sum()


# ------------------ Generation ------------------
def generate_coherent_response(
    input_text,
    model,
    word2idx,
    idx2word,
    max_length=30,
    temperature_local=temperature,
    top_p_local=top_p,
    repetition_penalty_local=repetition_penalty,
):
    model.eval()
    cleaned = clean_text(input_text)
    # tokenization: simple whitespace tokens (you can later replace with SentencePiece)
    tokens = cleaned.split() if cleaned else []
    tokens = tokens[-(max_len - 2) :]  # keep last tokens to stay within position embeddings
    src = text_to_tensor(tokens + [EOS_TOKEN], word2idx).unsqueeze(0)  # (1, S)
    tgt = torch.tensor([[word2idx[SOS_TOKEN]]], device=device)  # (1,1)
    generated_words = []
    last_tokens = []

    with torch.no_grad():
        for _ in range(max_length):
            src_padding_mask, tgt_padding_mask, tgt_mask = create_mask(src, tgt, word2idx)
            output = model(
                src,
                tgt,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                tgt_mask=tgt_mask,
            )  # (1, T, V)
            next_token_logits = output[:, -1, :]  # (1, V)
            # temperature
            next_token_logits = next_token_logits / max(1e-8, temperature_local)

            # repetition penalty: safer to subtract or scale logits for previous tokens
            for token_id in set(tgt[0].tolist() + last_tokens[-5:]):
                if token_id != word2idx[SOS_TOKEN]:
                    next_token_logits[0, token_id] = next_token_logits[0, token_id] / (repetition_penalty_local * 1.5)

            probs = torch.softmax(next_token_logits, dim=-1)
            probs = top_p_filter(probs, top_p_local)

            if temperature_local < 1e-4:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                if probs.sum() <= 0:
                    probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)  # (1,1)

            next_word = idx2word.get(int(next_token.item()), UNK_TOKEN)
            if next_word in (EOS_TOKEN, PAD_TOKEN, UNK_TOKEN):
                break
            if len(generated_words) > 3 and next_word in generated_words[-3:]:
                break

            tgt = torch.cat([tgt, next_token], dim=1)
            generated_words.append(next_word)
            last_tokens.append(int(next_token.item()))

            if int(next_token.item()) == word2idx[EOS_TOKEN]:
                break

    return " ".join(generated_words)


# ------------------ Training / Inference ------------------
if TRAIN_MODE:
    # training uses MongoDB if available (same as your earlier flow)
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
        questions = [clean_text(d.get("question", "")) for d in raw_data]
        answers = [clean_text(d.get("answer", "")) for d in raw_data]
    except Exception as e:
        # Fallback: try to load a local 'training_data.txt' file where each line is "question\tanswer"
        print(f"[warning] MongoDB not available or failed: {e}. Trying local fallback 'training_data.txt'...")
        if not os.path.exists("training_data.txt"):
            raise RuntimeError("Training data not found in MongoDB and no local 'training_data.txt' exists.")
        questions = []
        answers = []
        with open("training_data.txt", "r", encoding="utf-8") as f:
            for line in f:
                if "\t" in line:
                    q, a = line.strip().split("\t", 1)
                else:
                    parts = line.strip().split("||")
                    if len(parts) >= 2:
                        q, a = parts[0], parts[1]
                    else:
                        continue
                questions.append(clean_text(q))
                answers.append(clean_text(a))

    print(f"Loaded {len(questions)} training examples")

    # Build vocabulary with frequency thresholding to reduce vocab size
    counter = Counter()
    for q in questions:
        counter.update(q.split())
    for a in answers:
        counter.update(a.split())

    min_freq = 2  # keep tokens that appear at least twice
    kept_tokens = [w for w, c in counter.items() if c >= min_freq]

    # ensure special tokens present and placed deterministically
    special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    all_words = special_tokens + sorted([t for t in kept_tokens if t not in special_tokens])

    word2idx = {w: i for i, w in enumerate(all_words)}
    idx2word = {i: w for w, i in word2idx.items()}
    vocab_size = len(word2idx)

    print(f"Vocabulary size after filtering: {vocab_size}")
    print(f"Device: {device} | d_model: {d_model}")

    # Prepare tensors (use UNK fallback for OOVs)
    tensors_q = [text_to_tensor(q.split() + [EOS_TOKEN], word2idx) for q in questions]
    tensors_a = [text_to_tensor([SOS_TOKEN] + a.split() + [EOS_TOKEN], word2idx) for a in answers]

    # pad and truncate
    q_batch_all = pad_sequence(tensors_q, batch_first=True, padding_value=word2idx[PAD_TOKEN])
    a_batch_all = pad_sequence(tensors_a, batch_first=True, padding_value=word2idx[PAD_TOKEN])

    q_batch_all = q_batch_all[:, :max_len]
    a_batch_all = a_batch_all[:, :max_len]

    # model, optimizer, loss
    model = ImprovedTransformer(vocab_size, d_model, num_heads, num_layers, max_len, dropout=dropout_rate, padding_idx=word2idx[PAD_TOKEN]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx[PAD_TOKEN])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.8)
    epochs = 80
    min_loss = float("inf")
    patience_counter = 0
    early_stopping_patience = 12

    print("Starting training with improved transformer...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        perm = torch.randperm(q_batch_all.size(0))
        q_shuf = q_batch_all[perm]
        a_shuf = a_batch_all[perm]

        for i in range(0, len(q_shuf), batch_size):
            q_batch = q_shuf[i : i + batch_size].to(device)
            a_batch = a_shuf[i : i + batch_size].to(device)

            # apply word dropout to inputs (training augmentation)
            q_batch_aug = apply_word_dropout(q_batch, word2idx, rate=word_dropout_rate)
            a_in = apply_word_dropout(a_batch[:, :-1], word2idx, rate=word_dropout_rate)  # decoder input (without final token)

            src_padding_mask, tgt_padding_mask, tgt_mask = create_mask(q_batch_aug, a_in, word2idx)
            optimizer.zero_grad()
            output = model(
                q_batch_aug,
                a_in,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                tgt_mask=tgt_mask,
            )  # (B, T, V)

            loss = criterion(output.reshape(-1, vocab_size), a_batch[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        avg_loss = total_loss / (num_batches if num_batches > 0 else 1)
        scheduler.step(avg_loss)

        # checkpointing
        if avg_loss < min_loss:
            min_loss = avg_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "word2idx": word2idx,
                    "idx2word": idx2word,
                    "d_model": d_model,
                    "num_heads": num_heads,
                    "num_layers": num_layers,
                    "max_len": max_len,
                },
                "best_model.pth",
            )
        else:
            patience_counter += 1

        if (epoch + 1) % 1 == 0:
            print(f"[Epoch {epoch+1}/{epochs}] avg loss: {avg_loss:.4f}")
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

    # load best model after training if present
    if os.path.exists("best_model.pth"):
        ckpt = torch.load("best_model.pth", map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print("Best model loaded into memory (ready for inference).")

else:
    # inference: load checkpoint and start chat
    if not os.path.exists("best_model.pth"):
        raise FileNotFoundError("best_model.pth not found. Train first or provide checkpoint.")
    checkpoint = torch.load("best_model.pth", map_location=device)
    word2idx = checkpoint["word2idx"]
    idx2word = checkpoint["idx2word"]
    vocab_size = len(word2idx)

    model = ImprovedTransformer(vocab_size, d_model, num_heads, num_layers, max_len, dropout=dropout_rate, padding_idx=word2idx[PAD_TOKEN]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print("Loaded model + vocab from best_model.pth")

# ------------------ Chat loop (runs when TRAIN_MODE == False) ------------------
if not TRAIN_MODE:
    import gradio as gr

    # keep chat history in memory
    MAX_MEMORY_TURNS = 6
    chat_history = []

    def chat(user_input, history):
        global chat_history
        if user_input.strip().lower() == "!clear":
            chat_history = []
            return [], []

        chat_history.append(f"user: {clean_text(user_input)}")
        recent = chat_history[-MAX_MEMORY_TURNS:]
        context = " ".join(recent)
        ctx_tokens = clean_text(context).split()
        if len(ctx_tokens) > (max_len - 2):
            ctx_tokens = ctx_tokens[-(max_len - 2):]
        context = " ".join(ctx_tokens)

        response = generate_coherent_response(context, model, word2idx, idx2word, max_length=30)
        chat_history.append(f"lynqbit: {response}")

        if response:
            emojis = ["😼", "🐾", "⚡", "😹"]
            if random.random() > 0.2:
                response += " " + random.choice(emojis)
        else:
            response = "*processor purrs* Try asking me something else? 😼"

        # Append new message to history for chat UI
        history = history or []
        history.append((user_input, response))
        return history, history

    # Launch chat-style interface
    with gr.Blocks() as demo:
        gr.Markdown("## 😼 Lynqbit Chatbot")
        chatbox = gr.Chatbot(elem_id="chatbox", height=500)
        txt = gr.Textbox(show_label=False, placeholder="Type your message...", lines=2)
        txt.submit(chat, [txt, chatbox], [chatbox, chatbox])
        demo.launch(server_name="0.0.0.0", server_port=7860)
