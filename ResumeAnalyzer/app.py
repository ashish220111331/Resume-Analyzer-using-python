import streamlit as st
import pickle
import docx  # Extract text from Word file
import PyPDF2  # Extract text from PDF
import re
import numpy as np

# Load pre-trained model and TF-IDF vectorizer (ensure these are saved earlier)
svc_model = pickle.load(open('clf.pkl', 'rb'))  # Example file name, adjust as needed
tfidf = pickle.load(open('tfidf.pkl', 'rb'))  # Example file name, adjust as needed
le = pickle.load(open('encoder.pkl', 'rb'))  # Example file name, adjust as needed


# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText


# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text


# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    # Try using utf-8 encoding for reading the text file
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        # In case utf-8 fails, try 'latin-1' encoding as a fallback
        text = file.read().decode('latin-1')
    return text


# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text


# Function to predict the most relevant categories of a resume
def pred(input_resume, threshold=0.1):
    # Preprocess the input text (e.g., cleaning, etc.)
    cleaned_text = cleanResume(input_resume)

    # Vectorize the cleaned text using the same TF-IDF vectorizer used during training
    vectorized_text = tfidf.transform([cleaned_text])

    # Convert sparse matrix to dense
    vectorized_text = vectorized_text.toarray()

    # Get predicted probabilities for each category
    predictionProbs = svc_model.predict_proba(vectorized_text)[0]

    # Category Mapping (Your pre-defined categories)
    categoryMapping = {
        9: 'Data Science',
    18: 'HR',
    1: 'Advocate',
    2: 'Arts',
    34: 'Web Designing',
    22: 'Mechanical Engineer',
    30: 'Sales',
    20: 'Health and fitness',
    6: 'Civil Engineer',
    21: 'Java Developer',
    5: 'Business Analyst',
    29: 'SAP Developer',
    3: 'Automation Testing',
    15: 'Electrical Engineering',
    24: 'Operations Manager',
    27: 'Python Developer',
    11: 'DevOps Engineer',
    23: 'Network Security Engineer',
    25: 'PMO',
    10: 'Database',
    19: 'Hadoop',
    14: 'ETL Developer',
    13: 'DotNet Developer',
    4: 'Blockchain',
    31: 'Testing',
    0: 'AI Specialist',
    7: 'Cloud Architect',
    8: 'Cybersecurity Analyst',
    32: 'UI/UX Designer',
    26: 'Product Manager',
    12: 'Digital Marketing',
    28: 'Robotics Engineer',
    17: 'Game Developer',
    33: 'VR/AR Developer',
    16: 'Energy Analyst'
    }

    # Find the most likely category (category with the highest probability)
    most_likely_category_id = np.argmax(predictionProbs)
    most_likely_category = categoryMapping[most_likely_category_id]

    # Find all relevant categories based on the threshold
    relevantCategories = [
        categoryMapping[i] 
        for i, prob in enumerate(predictionProbs) 
        if prob > threshold
    ]

    return most_likely_category, relevantCategories


# Streamlit app layout
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")

    st.title("Resume Category Prediction App")
    st.markdown("Upload a resume in PDF, TXT, or DOCX format and get the predicted relevant job categories.")

    # File upload section
    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        # Extract text from the uploaded file
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.write("Successfully extracted the text from the uploaded resume.")

            # Display extracted text (optional)
            if st.checkbox("Show extracted text", False):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            # Add dynamic threshold slider
            threshold = st.slider("Select Threshold for Relevance", 0.0, 1.0, 0.1, 0.01)
            st.write(f"Using a threshold of: **{threshold}**")

            # Make prediction for multiple categories
            st.subheader("Prediction Results")

            # Call the prediction function
            most_likely_category, relevant_categories = pred(resume_text, threshold)

            # Display the most likely category
            st.write(f"Most likely category: **{most_likely_category}**")

            # Display all relevant categories based on the threshold
            if relevant_categories:
                st.write("Relevant categories based on the prediction probabilities:")
                for category in relevant_categories:
                    st.write(f"- **{category}**")
            else:
                st.write("No relevant categories found with the given threshold.")

        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")


if __name__ == "__main__":
    main()

