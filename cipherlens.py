import logging
import firebase_admin
from firebase_admin import credentials, db
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from datetime import datetime

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Firebase----------------------------------------------
cred = credentials.Certificate('firebasefile.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': ''
})

# Logging----------------------------------------------
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Bot Initialization----------------------------------------------
Token = ""
application = Application.builder().token(Token).build()

# Keywords
keywords = [
    "drug", "cocaine", "heroin", "meth", "methamphetamine", "weed", "marijuana", "hash", "hashish", "lsd",
    "ecstasy", "mdma", "narcotics", "opioid", "fentanyl", "crack", "shrooms", "psychedelics", "acid", "ketamine",
    "amphetamine", "benzo", "benzodiazepine", "xanax", "oxycontin", "adderall", "vicodin", "tramadol", "dmt",
    "salvia", "pcp", "synthetic", "designer drug", "smuggle", "trafficking", "cartel", "stash", "dealer", "selling",
    "buy", "purchase", "pill", "tablet", "powder", "inject", "snort", "dose", "trip", "overdose", "black market",
    "darknet", "dark web", "silk road", "illegal", "prescription", "street value", "illicit", "contraband",
    "shipment", "distribution", "laundering", "cash", "transaction", "crypto", "cryptocurrency", "bitcoin",
    "transfer", "shipment", "package", "delivery", "route", "customs", "hidden", "conceal", "disguise", "underground",
    "lab", "manufacture", "chemicals", "precursor", "equipment", "dissolve", "extract", "concentrate", "crystal",
    "rock", "liquid", "inhalant", "vape", "euphoria", "high", "kick", "buzz", "hallucinogen", "stimulant", "sedative",
    "numbing", "relaxant", "substance", "abuse", "addiction", "junkie", "pusher"
]



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



# New Chat Member Handling-------------------------------------
async def handle_new_chat_member(update: Update, context: ContextTypes.DEFAULT_TYPE):
    group_id = update.message.chat.id
    group_name = update.message.chat.title if update.message.chat.title else "Unnamed Group"
    logging.info(f"Storing group ID: {group_id} with name: {group_name} in the database")
    group_ref = db.reference(f'groups/{group_id}')
    group_data = group_ref.get()
    if not group_data:
        group_ref.set({
            'group_name': group_name,
            'admins': {},
            'flagged_messages': [],
            'no_of_flagged_messages': 0  # Add this line to initialize the count
        })
        logging.info(f"New group added: {group_id} with name: {group_name}")

# Get Group Name-------------------------------------
async def get_group_name(group_id):
    group_ref = db.reference(f'groups/{group_id}')
    group_data = group_ref.get()
    if group_data:
        return group_data.get('group_name')
    return None

# Start for Admin-------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    username = update.message.from_user.username
    group_id = update.message.chat.id
    group_ref = db.reference(f'groups/{group_id}')
    group_data = group_ref.get()

    if not group_data:
        await update.message.reply_text("Error: Group not found in the database. Please try again.")
        return

    admins_ref = db.reference(f'groups/{group_id}/admins')
    existing_admins = admins_ref.get()

    if existing_admins:
        await update.message.reply_text("There is already an admin for this group. Only one admin is allowed.")
    else:
        admin_ref = admins_ref.child(str(user_id)) 
        admin_ref.set({
            'user_id': user_id,
            'username': username
        })
        await update.message.reply_text("Welcome! You are now registered as the admin of this group.")

# Retrieve Admin ID-------------------------------------
async def get_admin_id():
    admin_ref = db.reference('admins/owner')
    admin_data = admin_ref.get()
    if admin_data:
        return admin_data.get('user_id')
    return None

# Verify User-------------------------------------
async def verify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[KeyboardButton("Share Phone Number", request_contact=True)]]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)
    await update.message.reply_text("Please share your phone number:", reply_markup=reply_markup)

# Update User in Database-------------------------------------
async def share_contact(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    user_id = user.id
    phone = update.message.contact.phone_number
    user_ref = db.reference(f'users/{user_id}')
    user_ref.set({
        'username': user.username,
        'full_name': user.full_name,
        'contact_number': phone,
        'is_authorized': True,
        'no_of_flagged_messages': 0  # Add this line to initialize the count
    })
    await context.bot.send_message(chat_id=user_id, text="You are now authorized to chat.")

# Initialize testing variable
testing = 0

# Initialize confusion matrix values
# Initialize confusion matrix values
confusion_matrix = {
    'TP': 0,
    'FP': 0,
    'TN': 0,
    'FN': 0
}

# New /testingmode command handler
async def testing_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global testing
    if testing == 1:
        keywords_list = ", ".join(keywords)  # Join the keywords into a string
        await update.message.reply_text(
            f"Testing mode is already active.\n"
            f"Keywords to test with: {keywords_list}\n"
            f"Try sending messages of 7 to 10 words.\n"
            f"Type 'yes' or 'no' for whether it is suspicious when the bot asks."
        )
        return  # Exit the function if testing mode is already active

    testing = 1
    keywords_list = ", ".join(keywords)  # Join the keywords into a string
    await update.message.reply_text(
        f"Testing mode activated. The bot will not process messages.\n"
        f"Keywords to test with: {keywords_list}\n"
        f"Try sending messages of 7 to 10 words.\n"
        f"Type 'yes' or 'no' for whether it is suspicious when the bot asks."
    )

# New /result command handler
# New /result command handler
async def result(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global testing
    testing = 0  # Deactivate testing mode

    # Retrieve the confusion matrix from the database
    confusion_matrix_ref = db.reference('testing')
    confusion_matrix = confusion_matrix_ref.get()

    if not confusion_matrix:
        await update.message.reply_text("No confusion matrix data found.")
        return

    # Extract values from the confusion matrix
    TP = confusion_matrix.get('TP', 0)
    FP = confusion_matrix.get('FP', 0)
    TN = confusion_matrix.get('TN', 0)
    FN = confusion_matrix.get('FN', 0)
    total_messages = TP + FP + TN + FN  # Calculate total messages

    # Calculate metrics
    accuracy = (TP + TN) / total_messages if total_messages > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    # Prepare the results message
    results_message = (
        f"Testing mode deactivated. Here are the results:\n"
        f"Confusion Matrix:\n"
        f"TP: {TP}\n"
        f"FP: {FP}\n"
        f"TN: {TN}\n"
        f"FN: {FN}\n"
        f"Total Messages Processed: {total_messages}\n\n"
        f"Metrics:\n"
        f"Accuracy: {accuracy:.2f}\n"
        f"Precision: {precision:.2f}\n"
        f"Recall: {recall:.2f}\n"
        f"F1 Score: {f1_score:.2f}\n"
    )

    # Send the results message to the chat
    await update.message.reply_text(results_message)

    # Optionally, reset the confusion matrix in the database for the next testing session
    db.reference('testing').set({
        'TP': 0,
        'FP': 0,
        'TN': 0,
        'FN': 0,
        'total_messages': 0  # Reset total messages as well
    })
# Updated echo function
# Initialize a variable to track if the bot is waiting for a confirmation response
waiting_for_confirmation = False
user_id_waiting = None  # To track which user is waiting for confirmation

# Updated echo function
async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global testing, waiting_for_confirmation, user_id_waiting, last_prediction
    user = update.message.from_user
    user_id = user.id
    user_message = update.message.text
    group_id = update.message.chat.id
    user_ref = db.reference(f'users/{user_id}')
    user_data = user_ref.get()

    if testing == 1:
        if waiting_for_confirmation and user_id == user_id_waiting:
            # Process user confirmation
            user_response = user_message.lower()

            # Determine if the last prediction was suspicious
            was_suspicious = last_prediction['was_suspicious']

            if user_response == 'yes':
                if was_suspicious:
                    confusion_matrix['TP'] += 1  # True Positive
                else:
                    confusion_matrix['FN'] += 1  # False Negative
            elif user_response == 'no':
                if was_suspicious:
                    confusion_matrix['FP'] += 1  # False Positive
                else:
                    confusion_matrix['TN'] += 1  # True Negative

            # Update the confusion matrix in the database
            db.reference('testing').set(confusion_matrix)

            # Acknowledge the user's response
            await update.message.reply_text("Thank you for your response! The confusion matrix has been updated.")
            waiting_for_confirmation = False  # Reset the waiting flag
            user_id_waiting = None  # Reset the waiting user ID
            return

        # Check for keywords in the user message
        if any(keyword in user_message.lower() for keyword in keywords):
            # Load the model and tokenizer
            model, tokenizer = load_model()
            label, suspicion_percentage = predict_message(model, tokenizer, user_message)

            # Determine if the message is suspicious
            is_suspicious = label == 1
            last_prediction = {'was_suspicious': is_suspicious}  # Store the prediction result

            if is_suspicious:
                response_message = f"⚠️ This message is suspicious! ⚠️\nSuspicion Percentage: {suspicion_percentage:.2f}%"
            else:
                response_message = f"✅ This message is not suspicious.\nSuspicion Percentage: {suspicion_percentage:.2f}%"

            # Send the response message
            await update.message.reply_text(response_message)

            # Ask for user confirmation
            await update.message.reply_text("Was this message really suspicious? Please reply with 'yes' or 'no'.")

            # Set the waiting flag and store the user ID
            waiting_for_confirmation = True
            user_id_waiting = user_id
        else:
            # No keywords found in the message
            await update.message.reply_text("No keyword present in the message.")
        return

    # Normal processing for non-testing mode
    if not user_data or not user_data.get("is_authorized"):
        await context.bot.delete_message(chat_id=update.message.chat.id, message_id=update.message.message_id)
        await context.bot.send_message(chat_id=update.message.chat.id, text=f"{user.first_name} Please Verify your Phone Number to chat.\nTo Verify, Check your DM.")
        await context.bot.send_message(chat_id=user_id, text=f"Type /verify to Verify your Phone Number.")
        return

    for keyword in keywords:
        if keyword in user_message.lower():
            # Call predict_message function
            model, tokenizer = load_model()  # Load the model and tokenizer
            label, suspicion_percentage = predict_message(model, tokenizer, user_message)

            # Check if the message is marked suspicious
            if label == 1 and suspicion_percentage > 94:  # Only flag if suspicion percentage is greater than 94%
                await flag_user(user, user_message, context, group_id, suspicion_percentage)
            break


# Flag Suspicious Messages-------------------------------------
async def flag_user(user, message, context, group_id, suspicion_percentage):
    user_id = user.id
    user_ref = db.reference(f'users/{user_id}')
    user_data = user_ref.get()
    contact_number = user_data.get('contact_number') if user_data else None
    group_name = await get_group_name(group_id)
    
    # Increment the number of flagged messages in the database
    group_ref = db.reference(f'groups/{group_id}')
    group_ref.child('no_of_flagged_messages').set(group_ref.child('no_of_flagged_messages').get() + 1)  # Increment the count by 1
    user_ref.child('no_of_flagged_messages').set(user_ref.child('no_of_flagged_messages').get() + 1)

    # Send message to admins
    admin_ids = db.reference(f'groups/{group_id}/admins').get().keys()
    for admin_id in admin_ids:
        admin_message = (
            f"⚠️ Suspicious Message Reported ⚠️\n"
            f"Group Name: {group_name}\n"
            f"Group ID: {group_id}\n"
            f"User   ID: {user_id}\n"
            f"Full Name: {user.full_name}\n"
            f"Username: {user.username}\n"
            f"Contact Number: {contact_number}\n"
            f"Message: {message}\n"
            f"Suspicion Percentage: {suspicion_percentage:.2f}%\n"
            f"Reason: Contains illegal keywords\n"
            f"Flagged At: {datetime.now().isoformat()}"
        )
        await context.bot.send_message(chat_id=admin_id, text=admin_message)

# Handlers for Telegram-------------------------------------
application.add_handler(CommandHandler("testingmode", testing_mode))  # Add the new command handler
application.add_handler(CommandHandler("result", result))  # Add the new result command handler
application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, handle_new_chat_member))
application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("verify", verify))
application.add_handler(MessageHandler(filters.CONTACT, share_contact))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))


# Run-------------------------------------
if __name__ == "__main__":
    application.run_polling()