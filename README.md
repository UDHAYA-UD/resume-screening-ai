# WEB-SCAN: AI-Powered Resume Screening System 🕸️

**Live Demo:** [https://resume-screening-model-ml.streamlit.app/](https://resume-screening-model-ml.streamlit.app/)

WEB-SCAN is a Streamlit-based web application that uses Natural Language Processing (NLP) and Machine Learning to automatically analyze resumes and predict the most suitable job role for a candidate. 

The application features a custom, dynamic Spider-Man theme and provides detailed visual insights into its predictions.

## 🚀 Features

- **Multiple Input Methods:** Upload a resume file (PDF or TXT) or paste the resume text directly into the app.
- **Smart Text Extraction:** Automatically extracts text from uploaded PDF files using `PyMuPDF`.
- **Advanced NLP Pipeline:** Uses `NLTK` to clean, tokenize, remove stop words, and lemmatize the resume text to prepare it for the model.
- **Machine Learning Prediction:** Employs a pre-trained **TF-IDF Vectorizer** and a **Logistic Regression** model to classify the resume into the most appropriate job category.
- **Confidence Breakdown:** Displays an interactive, horizontal bar chart (powered by `Plotly`) showing the model's confidence probability for the top predicted roles.

## 🧠 About the Model

The core of the screening system relies on two main components:
1. **TF-IDF Vectorizer (`tfidf_vectorizer.pkl`):** Converts the raw, preprocessed text into a numerical format by evaluating the Term Frequency-Inverse Document Frequency of the words.
2. **Logistic Regression Classifier (`resume_model.pkl`):** A machine learning model trained on a dataset of categorized resumes. It takes the vectorized text and predicts the probability of the resume belonging to various job roles.

### Preprocessing Steps
When a resume is submitted, the app performs the following preprocessing steps (identical to the training phase):
- Converts all text to lowercase.
- Tokenizes the text into individual words.
- Filters out non-alphabetic characters and common English stopwords.
- Lemmatizes the remaining words to their base forms (e.g., "running" becomes "run").

## 🛠️ Built With

- [Streamlit](https://streamlit.io/) - The web framework used.
- [Scikit-learn](https://scikit-learn.org/) - For the machine learning model (Logistic Regression & TF-IDF).
- [NLTK](https://www.nltk.org/) - For Natural Language Processing tasks.
- [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) - For PDF text extraction.
- [Plotly](https://plotly.com/python/) - For rendering the confidence charts.

## 💻 Local Installation

To run this project locally, clone the repository and install the dependencies:

```bash
git clone https://github.com/UDHAYA-UD/resume-screening-ai.git
cd resume-screening-ai
pip install -r requirements.txt
```

Then, start the Streamlit app:

```bash
streamlit run app.py
```
