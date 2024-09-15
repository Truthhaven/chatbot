import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def get_response(sentence):
    # Tokenize the user input
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Get model predictions
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    # Get the predicted tag
    tag = tags[predicted.item()]

    # Get the probability for the prediction
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # If probability is high enough, return the response
    if prob.item() > 0.75:
        # Loop through intents to find the matching tag
        for intent in intents['intents']:
            if tag == intent['tag']:
                # Return only the description of the first response
                response = intent['responses'][0]['description']
                return response
    else:
        return "I do not understand..."

# Main chat loop
print("Let's chat! (type 'quit' to exit)")
while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    # Get the response from the chatbot
    response = get_response(sentence)
    print(f"{bot_name}: {response}")
