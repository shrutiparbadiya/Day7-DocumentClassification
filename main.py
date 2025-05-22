import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import PyPDF2
import fitz  # PyMuPDF for rendering PDF pages as images
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from PIL import Image

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Load fine-tuned model and tokenizer
model_name = "./model/"  # Update with your saved model path if necessary
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
label_encoder = pickle.load(open("./model/label_encoder.pkl", 'rb'))

# Text cleaning function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text)  # Remove extra whitespace
    text = " ".join([stemmer.stem(word) for word in text.split() if word not in stop_words])  # Remove stopwords and stem
    return text

# Function for prediction
def predict(text):
    # Clean the input text
    text = clean_text(text)

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    return label_encoder.inverse_transform([predicted_class])[0]

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text

# Function to render PDF pages as images
def render_pdf(file):
    file.seek(0)  # Reset file pointer to the start
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    images = []
    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)
    return images

# Streamlit UI
st.title("Document Classifier with Transformer")
st.write("Upload a PDF file to classify its content into different categories.")

# Upload file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    pass
    # Extract text from PDF for classification
    text = extract_text_from_pdf(uploaded_file)

    # Predict class of the extracted text
    predicted_class = predict(text)

    # Display predicted class
    st.subheader("Predicted Class:")
    st.info(predicted_class)

    # Render PDF and display as images
    st.subheader("PDF Preview:")
    pdf_images = render_pdf(uploaded_file)
    for img in pdf_images:
        st.image(img, use_container_width=True)