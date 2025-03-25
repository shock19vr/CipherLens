import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F  # Import for softmax

# Define the model directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'drug_trafficking_model')

def load_model():
    """Load the trained model and tokenizer from the directory."""
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    return model, tokenizer

def predict_message(model, tokenizer, message):
    """Predict the label and suspicion percentage for a single message."""
    inputs = tokenizer(message, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1)
        label = probabilities.argmax().item()
        suspicion_percentage = probabilities[0, 1].item() * 100  # Get the percentage for class 1

    return label, suspicion_percentage

def main():
    """Main function to classify messages."""
    model, tokenizer = load_model()
    print("Model loaded. You can start entering messages to classify (Ctrl+C to exit).")
    
    while True:
        try:
            message = input("Enter a message: ")
            label, suspicion_percentage = predict_message(model, tokenizer, message)
            print(f"Message: '{message}' | Suspicious: {suspicion_percentage:.2f}% | Label: {label}")
        except KeyboardInterrupt:
            print("\nExiting the program.")
            break

if __name__ == "__main__":
    main()