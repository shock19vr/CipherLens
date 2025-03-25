import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import os
import matplotlib.pyplot as plt
import shutil
import torch.nn.functional as F
import tkinter as tk
from tkinter import ttk

# Define the model filename
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'drug_trafficking_model')

def load_data(csv_file):
    """Load the dataset from a CSV file."""
    df = pd.read_csv(csv_file)
    return df['message'], df['label']

def evaluate_model(trainer, test_dataset):
    """Evaluate the model on the test dataset."""
    predictions, labels, _ = trainer.predict(test_dataset)
    preds = predictions.argmax(axis=1)

    # Calculate softmax probabilities
    probabilities = F.softmax(torch.tensor(predictions), dim=1)

    # Get suspicion percentages for class 1 (suspicious)
    suspicion_percentages = probabilities[:, 1].numpy() * 100  # Convert to percentage

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    # Calculate TP, TN, FP, FN
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    # Calculate specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Calculate ROC AUC
    fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
    roc_auc = auc(fpr, tpr)

    print(f"Test set performance:\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    return accuracy, precision, recall, f1, suspicion_percentages, tp, tn, fp, fn, specificity, roc_auc, fpr, tpr, labels

def plot_loss_curves(trainer):
    """Plot training and validation loss curves."""
    history = trainer.state.log_history

    train_loss = [log['loss'] for log in history if 'loss' in log]
    eval_loss = [log['eval_loss'] for log in history if 'eval_loss' in log]

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(eval_loss, label='Validation Loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_accuracy_curves(trainer, test_dataset):
    """Plot accuracy curves."""
    predictions, labels, _ = trainer.predict(test_dataset)
    preds = predictions.argmax(axis=1)
    accuracy = accuracy_score(labels, preds)

    plt.figure(figsize=(10, 5))
    plt.plot([0, 1], [accuracy, accuracy], label='Accuracy', color='green')
    plt.title('Accuracy Curve')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_precision_recall_curve(labels, probabilities):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(labels, probabilities[:, 1])
    plt.figure(figsize=(10, 5))
    plt.plot(recall, precision, marker='.')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()

def plot_roc_curve(fpr, tpr, roc_auc):
    """Plot ROC curve."""
    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

def plot_confusion_matrix(labels, preds):
    """Plot confusion matrix."""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix plt.colorbar()')
    tick_marks = range(len(set(labels)))
    plt.xticks(tick_marks, ['Not Suspicious', 'Suspicious'])
    plt.yticks(tick_marks, ['Not Suspicious', 'Suspicious'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def check_dataset_size(labels):
    """Check the dataset size and class distribution."""
    class_counts = labels.value_counts()
    print("Class distribution:\n", class_counts)

    if len(labels) < 100:
        print("Warning: The dataset is too small.")
    if class_counts.min() < 10:
        print("Warning: The dataset may be unbalanced.")

def delete_existing_model(model_dir):
    """Delete the existing model directory if it exists."""
    if os.path.exists(model_dir):
        try:
            shutil.rmtree(model_dir)
            print(f"Deleted existing model directory: {model_dir}")
        except Exception as e:
            print(f"Error deleting existing model directory: {e}")

def train_model(texts, labels):
    """Train the Hugging Face model on the provided dataset."""
    delete_existing_model(MODEL_DIR)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=512)

    class DrugTraffickingDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    dataset = DrugTraffickingDataset(encodings, labels.tolist())

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        eval_strategy="epoch",
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()

    # Evaluate the model and get suspicion percentages
    accuracy, precision, recall, f1, suspicion_percentages, tp, tn, fp, fn, specificity, roc_auc, fpr, tpr, test_labels = evaluate_model(trainer, test_dataset)

    # Show results in GUI
    show_results(accuracy, precision, recall, f1 , tp, tn, fp, fn, suspicion_percentages, specificity, roc_auc, fpr, tpr, test_labels)

    # Plot loss curves
    plot_loss_curves(trainer)

    # Save the model and tokenizer
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    return accuracy, precision, recall, f1, suspicion_percentages, tp, tn, fp, fn

def show_results(accuracy, precision, recall, f1, tp, tn, fp, fn, suspicion_percentages, specificity, roc_auc, fpr, tpr, labels):
    """Display results in a GUI."""
    root = tk.Tk()
    root.title("Model Evaluation Results")

    # Display metrics
    metrics = f"""
    Accuracy: {accuracy:.4f}
    Precision: {precision:.4f}
    Recall: {recall:.4f}
    F1 Score: {f1:.4f}
    Specificity: {specificity:.4f}
    AUC-ROC Score: {roc_auc:.4f}
    
    TP: {tp}
    TN: {tn}
    FP: {fp}
    FN: {fn}
    """
    label = tk.Label(root, text=metrics)
    label.pack(pady=10)

    # Plot suspicion percentages
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(suspicion_percentages)), suspicion_percentages)
    plt.title('Suspicion Percentages for Class 1 (Suspicious)')
    plt.xlabel('Sample Index')
    plt.ylabel('Suspicion Percentage (%)')
    plt.show()

    # Plot ROC curve
    plot_roc_curve(fpr, tpr, roc_auc)

    # Convert suspicion percentages to binary predictions
    preds = [1 if p > 0.5 else 0 for p in suspicion_percentages]  # Convert probabilities to binary predictions

    # Plot confusion matrix
    plot_confusion_matrix(labels, preds)

    # Show the GUI
    root.mainloop()

def main():
    """Main function to execute the training process."""
    csv_file = 'dataset.csv'  # Updated path to your dataset
    texts, labels = load_data(csv_file)
    check_dataset_size(labels)
    train_model(texts, labels)

if __name__ == "__main__":
    main()