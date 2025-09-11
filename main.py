# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from torch.nn.utils.rnn import pad_sequence
import random

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
hidden_size = 512 # admin
teacher_forcing_ratio = 0.7
top_k = 5  # top-k sampling for varied output

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------ Seq2Seq Model ------------------
class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.decoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, src, tgt=None, max_len=80):
        # Encoder
        embedded = self.embedding(src)
        _, hidden = self.encoder(embedded)

        # Decoder
        batch_size = src.size(0)
        input_dec = torch.tensor([[word2idx[SOS_TOKEN]]] * batch_size, device=device)
        outputs = []

        for t in range(max_len):
            emb_dec = self.embedding(input_dec)
            out, hidden = self.decoder(emb_dec, hidden)
            pred = self.fc(out.squeeze(1))
            outputs.append(pred.unsqueeze(1))

            # Teacher forcing
            if tgt is not None and t < tgt.size(1) and random.random() < teacher_forcing_ratio:
                input_dec = tgt[:, t].unsqueeze(1)
            else:
                input_dec = torch.argmax(pred, dim=1).unsqueeze(1)

        return torch.cat(outputs, dim=1)

# ------------------ Helpers ------------------
def text_to_tensor(words):
    idxs = [word2idx.get(w, word2idx[PAD_TOKEN]) for w in words]
    return torch.tensor(idxs, dtype=torch.long, device=device)

def tensor_to_text(tensor):
    return " ".join([idx2word[i.item()] for i in tensor if idx2word[i.item()] not in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]])

# ------------------ Prepare batches ------------------
tensors_q = [text_to_tensor(q + [EOS_TOKEN]) for q in questions]
tensors_a = [text_to_tensor([SOS_TOKEN] + a + [EOS_TOKEN]) for a in answers]

q_batch = pad_sequence(tensors_q, batch_first=True, padding_value=word2idx[PAD_TOKEN])
a_batch = pad_sequence(tensors_a, batch_first=True, padding_value=word2idx[PAD_TOKEN])

# Move data to device
q_batch = q_batch.to(device)
a_batch = a_batch.to(device)

# ------------------ Training ------------------
model = Seq2Seq(vocab_size, hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002)
criterion = nn.CrossEntropyLoss(ignore_index=word2idx[PAD_TOKEN])

epochs = 400
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(q_batch, a_batch, max_len=a_batch.size(1))
    
    # Reshape for loss calculation
    output = output.view(-1, vocab_size)
    target = a_batch.view(-1)
    
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

print("\n😼 Lynqbit is online! Type your message (or !exit to quit)\n")

# ------------------ Chat Loop with top-k sampling ------------------
model.eval()
while True:
    user_input = input("You: ").lower()
    if user_input == "!exit":
        break
        
    words = user_input.split()
    q_tensor = text_to_tensor(words + [EOS_TOKEN]).unsqueeze(0)
    
    with torch.no_grad():
        # Encoder
        embedded = model.embedding(q_tensor)
        _, hidden = model.encoder(embedded)
        
        # Decoder
        input_dec = torch.tensor([[word2idx[SOS_TOKEN]]], device=device)
        pred_words = []

        for _ in range(50):  # max words
            emb_dec = model.embedding(input_dec)
            out, hidden = model.decoder(emb_dec, hidden)
            pred = model.fc(out.squeeze(1))
            probs = torch.softmax(pred, dim=1)

            # top-k sampling
            values, indices = torch.topk(probs, top_k)
            next_word_idx = indices[0, torch.multinomial(values[0], 1)].item()

            if idx2word[next_word_idx] == EOS_TOKEN:
                break

            pred_words.append(next_word_idx)
            input_dec = torch.tensor([[next_word_idx]], device=device)

        # After generating indices
        reply = " ".join([idx2word[i] for i in pred_words if idx2word[i] not in [SOS_TOKEN, EOS_TOKEN]])
        print(f"Lynqbit: {reply}")