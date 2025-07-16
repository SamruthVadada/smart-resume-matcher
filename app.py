import gradio as gr
import pandas as pd
import numpy as np
import os
import re
import tempfile
from docx import Document
from PyMuPDF import fitz # PyMuPDF is imported as fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model 'en_core_web_sm' loaded.")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    print("Attempting to download and load 'en_core_web_sm'...")
    try:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        print("✔ Download and installation successful")
        print("✅ spaCy model 'en_core_web_sm' downloaded and loaded.")
    except Exception as download_e:
        print(f"❌ Failed to download and load spaCy model: {download_e}")
        # Fallback for nlp if download fails, though core functionality will be impacted
        nlp = None

print("✅ All core libraries imported.")

# --- Utility Functions ---
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        return f"Error extracting text from PDF: {e}"
    return text

def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file."""
    text = ""
    try:
        doc = Document(docx_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        return f"Error extracting text from DOCX: {e}"
    return text

def is_resume(file_content, file_name):
    """
    Heuristically checks if the content and filename suggest it's a resume.
    This is a simple check and can be improved.
    """
    if not file_content:
        return False

    # Check file extension
    if file_name and not (file_name.lower().endswith(('.pdf', '.docx'))):
        return False

    # Keywords often found in resumes
    resume_keywords = [
        "experience", "education", "skills", "summary", "profile",
        "work history", "projects", "achievements", "qualifications",
        "resume", "cv"
    ]
    if not any(keyword in file_content.lower() for keyword in resume_keywords):
        return False

    # Check for common non-resume indicators (e.g., very short content, specific phrases)
    if len(file_content.split()) < 50: # Too short to be a resume
        return False

    return True

print("✅ Unified utility functions defined (with further refined resume detection).")

# --- Job Seeker Mode Data ---
job_roles = {
    "Data Scientist": "Data Scientist with strong statistical modeling, machine learning, and programming skills in Python/R. Experience with data visualization tools like Tableau/PowerBI and big data technologies (Spark, Hadoop).",
    "Software Engineer": "Software Engineer proficient in developing scalable applications using Java/Python/C++. Strong understanding of data structures, algorithms, and software design principles. Experience with cloud platforms (AWS, Azure, GCP) and agile methodologies.",
    "UX Designer": "UX Designer passionate about creating user-centered designs. Expertise in user research, wireframing, prototyping (Figma, Sketch, Adobe XD), and usability testing. Strong portfolio demonstrating design thinking and problem-solving skills.",
    "Marketing Manager": "Marketing Manager with proven experience in developing and executing digital marketing campaigns, SEO/SEM, social media, and content strategy. Strong analytical skills and ability to drive brand awareness and lead generation.",
    "Project Manager": "Project Manager with experience leading cross-functional teams, managing project lifecycles, and delivering projects on time and within budget. Proficient in Agile/Scrum methodologies and project management tools (Jira, Asana)."
}

# Initialize TF-IDF Vectorizer for Job Seeker Mode
# This vectorizer will be fitted once with all job descriptions
all_job_descriptions_text = list(job_roles.values())
job_desc_vectorizer = TfidfVectorizer(stop_words='english')
job_desc_vectorizer.fit(all_job_descriptions_text)

print("✅ Job roles loaded and Job Seeker TF-IDF vectorizer initialized and fitted.")

# --- Core Matching Logic ---
def get_cosine_similarity(text1, text2, vectorizer):
    """Calculates cosine similarity between two texts using a fitted TF-IDF vectorizer."""
    if not text1 or not text2:
        return 0.0
    tfidf_matrix = vectorizer.transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def match_resumes_to_job_description(job_description, resume_files):
    """
    Matches multiple resumes to a single job description.
    Returns a DataFrame with scores and a CSV file path.
    """
    if not job_description:
        return "Please provide a job description.", None

    if not resume_files:
        return "Please upload at least one resume file.", None

    # Use a new TF-IDF vectorizer for recruiter mode to fit all resumes + JD
    corpus = [job_description]
    for resume_file in resume_files:
        temp_file_path = resume_file.name
        file_extension = os.path.splitext(temp_file_path)[1].lower()
        resume_text = ""

        if file_extension == '.pdf':
            resume_text = extract_text_from_pdf(temp_file_path)
        elif file_extension == '.docx':
            resume_text = extract_text_from_docx(temp_file_path)
        else:
            # Skip non-PDF/DOCX files
            continue

        if not is_resume(resume_text, resume_file.name):
            print(f"Skipping {resume_file.name}: Not detected as a valid resume or empty content.")
            continue

        corpus.append(resume_text)

    if len(corpus) <= 1: # Only job description or no valid resumes found
        return "No valid resumes found to process.", None

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)

    job_desc_vector = tfidf_matrix[0:1] # First item is the job description
    resume_vectors = tfidf_matrix[1:]   # Remaining items are resumes

    similarities = cosine_similarity(job_desc_vector, resume_vectors)[0]

    results = []
    valid_resume_names = [os.path.basename(f.name) for f in resume_files if is_resume(extract_text_from_pdf(f.name) if os.path.splitext(f.name)[1].lower() == '.pdf' else extract_text_from_docx(f.name), f.name)]
    
    # Ensure similarities and valid_resume_names match in length
    if len(similarities) != len(valid_resume_names):
        # This case should ideally not happen if is_resume filtering is consistent
        # For robustness, we'll try to align them or return an error.
        # A more robust solution would be to filter corpus based on is_resume
        # before vectorizing, but for now, we'll proceed with a warning.
        print("Warning: Mismatch between number of similarities and valid resume names.")
        # Re-aligning by re-creating corpus with only valid resumes for safety
        filtered_corpus = [job_description]
        filtered_resume_files = []
        for resume_file in resume_files:
            temp_file_path = resume_file.name
            file_extension = os.path.splitext(temp_file_path)[1].lower()
            resume_text = ""
            if file_extension == '.pdf':
                resume_text = extract_text_from_pdf(temp_file_path)
            elif file_extension == '.docx':
                resume_text = extract_text_from_docx(temp_file_path)
            
            if is_resume(resume_text, resume_file.name):
                filtered_corpus.append(resume_text)
                filtered_resume_files.append(resume_file)
        
        if len(filtered_corpus) <= 1:
            return "No valid resumes found to process after re-filtering.", None

        vectorizer_filtered = TfidfVectorizer(stop_words='english')
        tfidf_matrix_filtered = vectorizer_filtered.fit_transform(filtered_corpus)
        job_desc_vector_filtered = tfidf_matrix_filtered[0:1]
        resume_vectors_filtered = tfidf_matrix_filtered[1:]
        similarities = cosine_similarity(job_desc_vector_filtered, resume_vectors_filtered)[0]
        valid_resume_names = [os.path.basename(f.name) for f in filtered_resume_files]


    for i, score in enumerate(similarities):
        results.append({
            "Resume": valid_resume_names[i],
            "Match Score": f"{score * 100:.2f}%"
        })

    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by="Match Score", ascending=False)

    # Create a temporary CSV file
    csv_path = os.path.join(tempfile.gettempdir(), "resume_match_scores.csv")
    df_sorted.to_csv(csv_path, index=False)

    return df_sorted, csv_path

def get_job_seeker_match(resume_file, job_description_text, predefined_jd_key):
    """
    Calculates a single match score for a job seeker.
    """
    if predefined_jd_key and predefined_jd_key != "Custom":
        job_description_text = job_roles.get(predefined_jd_key, "")
    
    if not job_description_text:
        return "Please provide a job description (either custom or select a predefined role).", None

    if not resume_file:
        return "Please upload your resume.", None

    temp_file_path = resume_file.name
    file_extension = os.path.splitext(temp_file_path)[1].lower()
    resume_text = ""

    if file_extension == '.pdf':
        resume_text = extract_text_from_pdf(temp_file_path)
    elif file_extension == '.docx':
        resume_text = extract_text_from_docx(temp_file_path)
    else:
        return "Unsupported resume file format. Please upload a PDF or DOCX.", None

    if not is_resume(resume_text, resume_file.name):
        return "The uploaded file does not appear to be a valid resume or is empty. Please upload a proper resume (PDF/DOCX).", None

    # Use the pre-fitted job_desc_vectorizer
    # Add resume text to the corpus for transformation
    corpus_for_match = [job_description_text, resume_text]
    
    # Transform the new corpus using the pre-fitted vectorizer
    # Note: This will only use features seen during the initial fit
    # For job seeker mode, this is acceptable as the job descriptions are fixed.
    # If custom JDs were to introduce entirely new vocabulary, a re-fit might be needed
    # but for typical use cases, it's fine.
    tfidf_matrix = job_desc_vectorizer.transform(corpus_for_match)
    
    job_desc_vector = tfidf_matrix[0:1]
    resume_vector = tfidf_matrix[1:2]

    similarity = cosine_similarity(job_desc_vector, resume_vector)[0][0]
    match_score = f"{similarity * 100:.2f}%"
    
    return f"Your Resume Match Score: {match_score}", None

print("✅ Recruiter matching logic defined.")
print("✅ Job Seeker matching logic defined.")

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# Smart Resume Matcher 🧠📄")
    gr.Markdown("An AI-powered tool for matching resumes to job descriptions, with modes for both recruiters and job seekers.")

    with gr.Tabs():
        with gr.TabItem("Recruiter Mode"):
            gr.Markdown("## Recruiters: Find Your Top Candidates!")
            gr.Markdown("Upload multiple resumes (PDF/DOCX) and a single job description to rank candidates by match score.")
            
            with gr.Row():
                recruiter_jd_input = gr.Textbox(label="Job Description (for Recruiters)", lines=5, placeholder="Paste your job description here...")
                recruiter_resume_files = gr.File(label="Upload Resumes (PDF/DOCX)", file_count="multiple", file_types=[".pdf", ".docx"])
            
            recruiter_output_df = gr.DataFrame(headers=["Resume", "Match Score"], label="Resume Match Results")
            recruiter_download_btn = gr.File(label="Download Results CSV", file_count="single", interactive=False)
            
            recruiter_submit_btn = gr.Button("Match Resumes")
            recruiter_submit_btn.click(
                fn=match_resumes_to_job_description,
                inputs=[recruiter_jd_input, recruiter_resume_files],
                outputs=[recruiter_output_df, recruiter_download_btn]
            )

        with gr.TabItem("Job Seeker Mode"):
            gr.Markdown("## Job Seekers: Optimize Your Resume!")
            gr.Markdown("Upload your resume (PDF/DOCX) and get an instant match score against a job description.")
            
            with gr.Row():
                job_seeker_resume_file = gr.File(label="Upload Your Resume (PDF/DOCX)", file_count="single", file_types=[".pdf", ".docx"])
                with gr.Column():
                    predefined_jd_dropdown = gr.Dropdown(
                        label="Select a Predefined Job Role (Optional)",
                        choices=list(job_roles.keys()) + ["Custom"],
                        value="Custom"
                    )
                    job_seeker_jd_input = gr.Textbox(label="Custom Job Description (for Job Seekers)", lines=5, placeholder="Paste your job description here...")
            
            job_seeker_output_text = gr.Textbox(label="Your Match Score", interactive=False)
            
            # Helper function to clear custom JD if predefined is selected
            def update_jd_input(choice):
                if choice != "Custom":
                    return gr.Textbox(value=job_roles.get(choice, ""), interactive=False)
                else:
                    return gr.Textbox(value="", interactive=True)

            predefined_jd_dropdown.change(
                fn=update_jd_input,
                inputs=predefined_jd_dropdown,
                outputs=job_seeker_jd_input
            )

            job_seeker_submit_btn = gr.Button("Get Match Score")
            job_seeker_submit_btn.click(
                fn=get_job_seeker_match,
                inputs=[job_seeker_resume_file, job_seeker_jd_input, predefined_jd_dropdown],
                outputs=job_seeker_output_text
            )

print("✅ Gradio UI updated with Job Seeker custom JD soft warning.")

if __name__ == "__main__":
    # Get the port from the environment variable, default to 7860 if not set (for local testing)
    port = int(os.environ.get("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=port)
