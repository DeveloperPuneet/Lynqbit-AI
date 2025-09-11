from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI) # Connect to MongoDB

# Database + Collection
db = client["lynqbit_db"]
collection = db["training_data"] # Access collection

def add_training_data(trainer, question, answer):
    data = { # Create data dict
        "trainer": trainer,
        "question": question.lower().strip(), # Lowercase, strip
        "answer": answer.lower().strip() # Lowercase, strip
    }
    collection.insert_one(data) # Insert data
    print(f"[✅] Added training pair by {trainer}: ({question} → {answer})") # Print success msg

if __name__ == "__main__":
    print("😼 Lynqbit Trainer Mode (type !exit to stop)\n") # Intro message
    trainer_name = input("Enter your trainer name: ").strip() # Get trainer name

    while True: # Start training loop
        q = input("Question: ").strip() # Get the question
        if q == "!exit": # Check exit condition
            print("Exiting Trainer Mode...") # Exit message
            break
        a = input("Answer: ").strip() # Get the answer
        add_training_data(trainer_name, q, a) # Add training data