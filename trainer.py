from pymongo import MongoClient
from dotenv import load_dotenv
import json
import os
import sys

# Load .env file
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)  # Connect to MongoDB

# Database + Collection
db = client["lynqbit_db"]
collection = db["training_data"]  # Access collection

def add_training_data(trainer, question, answer):
    data = {  # Create data dict
        "trainer": trainer,
        "question": question.lower().strip(),  # Lowercase, strip
        "answer": answer.lower().strip()  # Lowercase, strip
    }
    collection.insert_one(data)  # Insert data
    print(f"[✅] Added training pair by {trainer}: ({question} → {answer})")  # Print success msg

def process_json_file(file_path, trainer_name):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            for item in data:
                question = item.get("instruction", "").strip()
                answer = item.get("output", "").strip()
                
                if question and answer:  # Only add if both fields exist
                    add_training_data(trainer_name, question, answer)
                else:
                    print(f"[⚠️] Skipped item due to missing instruction/output: {item}")
                    
        print(f"[✅] Successfully processed file: {file_path}")
        return True
        
    except FileNotFoundError:
        print(f"[❌] File not found: {file_path}")
        return False
    except json.JSONDecodeError:
        print(f"[❌] Invalid JSON format in file: {file_path}")
        return False
    except Exception as e:
        print(f"[❌] Error processing file: {str(e)}")
        return False

if __name__ == "__main__":
    # Check if alpaca_data.json exists in the same directory
    json_file = "alpaca_data.json"
    
    if os.path.exists(json_file):
        response = input(f"Found '{json_file}'. Do you want to import it? (y/n): ").strip().lower()
        if response == 'y' or response == 'yes':
            trainer_name = "system"  # Default trainer name for file imports
            process_json_file(json_file, trainer_name)
            print("\n" + "="*50 + "\n")
    
    # Interactive mode
    print("😼 Lynqbit Trainer Mode (type !exit to stop)\n")
    trainer_name = input("Enter your trainer name: ").strip()

    while True:
        q = input("Question: ").strip()
        if q == "!exit":
            print("Exiting Trainer Mode...")
            break
        a = input("Answer: ").strip()
        add_training_data(trainer_name, q, a)
        