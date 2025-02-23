import streamlit as st
from datasets import load_dataset
from transformers import BertTokenizer
import torch
from huggingface_hub import hf_hub_download, login
from Siames import SiameseNetworkWithBERT  # assuming you have this class defined

# Tokenizer setup
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Login to Hugging Face
login("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")

# Dataset loading
dataset = load_dataset("multi_nli")
dataset = dataset.shuffle()
train_data = dataset['train'].select(range(100))  
validation_data = dataset['validation_matched'].select(range(20))

# Caching model loading function without cache_resource
def load_model():
    model_id = "Aman010/Bert-Siames"
    
    # Try downloading the model weights and handle exceptions
    try:
        model_path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
        pretrained_model = torch.load(model_path, weights_only=False)
        pretrained_model_state_dict = pretrained_model.state_dict()
        
        st.success('Model loaded successfully!')
        
        # Instantiate your model class (assuming it's defined)
        model = SiameseNetworkWithBERT(num_labels=3)  # Make sure your model class is defined
        model_state_dict = model.state_dict()

        # Load weights into your model
        for name, param in pretrained_model_state_dict.items():
            if name in model_state_dict:
                model_state_dict[name].copy_(param)  # Copy weights if name matches
        
        model.load_state_dict(model_state_dict)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Load the model at startup
model = load_model()

def inference(model, tokenizer, sentence1, sentence2, device):
    # Tokenize the input sentences
    inputs = tokenizer(sentence1, sentence2, padding=True, truncation=True, return_tensors="pt", max_length=128)
    input_ids1 = inputs['input_ids'].detach().numpy()
    attention_mask1 = inputs['attention_mask'].detach().numpy()
    
    # Inference
    model.eval()
    with torch.no_grad():
        logits = model(input_ids1, attention_mask1)  # Ensure your model's forward function is correct
    
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = torch.max(probabilities, dim=1)
    return predicted_class, probabilities

# Inference logic
if st.button("Predict"):
    input1 = st.text_input("Enter Text 1", placeholder="First input text")
    input2 = st.text_input("Enter Text 2", placeholder="Second input text")
    
    if input1 and input2:
        out = inference(model, tokenizer, input1, input2, "cpu")  # Adjust device if needed
        st.write(f"Predicted Class: {out[0][1]}")
        st.bar_chart(out[1].detach().numpy().flatten())  # Plot probabilities
    else:
        st.warning("Please enter both texts to predict.")
