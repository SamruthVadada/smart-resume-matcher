# app.py - Main application script for Smart Resume Matcher

import gradio as gr
import pandas as pd
import numpy as np
import string
import re
import fitz  # PyMuPDF for PDF extraction
from docx import Document  # For DOCX extraction
import os  # For os.path.basename
import tempfile  # For creating temporary files for CSV download
import spacy # Import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("✅ All core libraries imported.")

# --- SpaCy Model Download Logic ---
# This block ensures the 'en_core_web_sm' model is downloaded if it's not already available.
# It's crucial for CPU-only environments like Hugging Face Spaces where models aren't pre-installed.
try:
    # Try to load the model if it's already downloaded
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model 'en_core_web_sm' loaded.")
except OSError:
    # If not, download it using spaCy's CLI
    print("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm") # Load it after successful download
    print("✅ spaCy model 'en_core_web_sm' downloaded and loaded.")

# --- Unified Utility Functions ---

def clean_text(text):
    """
    Cleans text by converting to lowercase, removing punctuation, digits, and extra spaces.

    Args:
        text (str): The input string to be cleaned.

    Returns:
        str: The cleaned string. Returns an empty string if input is not a string.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)  # Removes all digits
    text = re.sub(r'\s+', ' ', text).strip()  # Removes extra spaces
    return text


def extract_text_from_file(file_path):
    """
    Extracts text content from a PDF or DOCX file path.

    Args:
        file_path (str): The path to the PDF or DOCX file.

    Returns:
        str: The extracted text content, or an error message if extraction fails or file type is unsupported.
    """
    try:
        if file_path.endswith(".pdf"):
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text

        elif file_path.endswith(".docx"):
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])

        else:
            return "❌ Unsupported file type. Please upload PDF or DOCX."
    except Exception as e:
        return f"❌ Error extracting text from {os.path.basename(file_path)}: {str(e)}"


def is_probably_resume(text):
    """
    Refined check for whether the uploaded text is likely a resume.
    Prioritizes strong negative indicators for non-resumes and simpler positive indicators for resumes.

    Args:
        text (str): The text content extracted from a document.

    Returns:
        bool: True if the text is likely a resume, False otherwise.
    """
    text_lower = text.lower()

    # 1. Basic length check: A proper resume usually has a decent amount of text.
    # Filters out very short, non-substantive files.
    if len(text.strip()) < 200:
        return False

    # 2. Strong Negative Indicators (Keywords/phrases highly common in non-resume documents)
    # If a document contains multiple strong indicators of being a research paper or certificate,
    # it's very likely NOT a resume.
    research_paper_indicators = [
        "abstract", "introduction", "conclusion", "references", "bibliography",
        "doi:", "figure ", "table ", "methodology", "discussion",
        "literature review", "journal of", "peer-reviewed", "academic paper",
        "copyright ©", "volume ", "issue ", "acknowledgements", "citation",
        "author guidelines", "submission guidelines", "thesis", "dissertation"
    ]
    certificate_indicators = [
        "certificate of completion", "awarded to", "issued by", "for successfully completing",
        "achieved a score of", "exam date", "enrollment date", "certification id", "granted by",
        "has successfully completed", "date of issue", "serial number", "credential id",
        "course instructor", "learning hours", "passing score", "signature of issuing body"
    ]

    # Count occurrences of negative indicators
    num_research_indicators = sum(1 for indicator in research_paper_indicators if indicator in text_lower)
    num_certificate_indicators = sum(1 for indicator in certificate_indicators if indicator in text_lower)

    # If there are multiple strong negative indicators, classify as non-resume.
    # Adjusted thresholds: 3+ for research, 2+ for certificates are strong signals.
    if num_research_indicators >= 3 or num_certificate_indicators >= 2:
        return False

    # 3. Positive Indicators (Keywords/phrases generally found in resumes)
    # These are general indicators. A resume should have at least some of these.
    positive_keywords = [
        "experience", "skills", "education", "projects", "summary", "objective",
        "work history", "employment", "professional experience", "qualifications",
        "certifications", "achievements", "awards", "volunteer", "leadership",
        "contact information", "phone", "email", "linkedin", "github", "profile"
    ]

    # Count how many general resume keywords are present
    num_positive_keywords = sum(1 for keyword in positive_keywords if keyword in text_lower)

    # A document is probably a resume if it contains a reasonable number of common resume terms
    # and has not been flagged by the strong negative indicators.
    # Lowered threshold slightly to be more inclusive for valid resumes.
    return num_positive_keywords >= 3

print("✅ Unified utility functions defined (with further refined resume detection).")


# --- Job Descriptions & Job Seeker TF-IDF Setup ---
# Dictionary of job roles and their detailed job descriptions.
# These descriptions serve as the core reference for matching in both Recruiter and Job Seeker modes.
job_roles = {
    "Data Analyst": """
Job Title: Data Analyst

Responsibilities:
- Analyze large datasets to identify meaningful insights and trends
- Build dashboards and data visualizations using Power BI, Excel, or Tableau
- Collaborate with stakeholders to gather requirements and translate them into reports
- Clean and preprocess raw data from multiple sources
- Present findings and recommendations to business leaders

Required Skills:
- Strong SQL and Excel skills
- Proficiency in Python (Pandas, NumPy, Matplotlib)
- Experience with Tableau or Power BI
- Data cleaning and transformation techniques
- Strong communication and storytelling ability

Bonus:
- Experience with predictive modeling or machine learning
- Exposure to cloud platforms like AWS, BigQuery, or Snowflake
""",

    "Web Developer": """
Job Title: Web Developer

Responsibilities:
- Build responsive websites using HTML, CSS, and JavaScript
- Use frontend frameworks like React, Angular, or Vue.js
- Optimize web applications for speed and scalability
- Debug and maintain existing websites
- Collaborate with designers and backend developers

Required Skills:
- Strong knowledge of HTML5, CSS3, and JavaScript
- Experience with React, Vue, or Angular
- Understanding of responsive design
- Familiarity with REST APIs and version control (Git)

Bonus:
- Knowledge of UI/UX principles
- Experience with tools like Webpack, NPM, and hosting platforms
""",

    "Software Engineer (SDE)": """
Job Title: Software Development Engineer (SDE)

Responsibilities:
- Design and develop scalable, high-performance software solutions
- Participate in code reviews and software testing
- Collaborate with product and design teams
- Debug and resolve technical issues
- Follow software engineering best practices

Required Skills:
- Proficiency in Java, C++, Python, or other OOP languages
- Strong foundation in data structures and algorithms
- Familiarity with system design and architecture
- Understanding of databases and APIs

Bonus:
- Experience with cloud platforms (AWS, GCP, Azure)
- Familiarity with CI/CD tools and containers (Docker, Kubernetes)
""",

    "Machine Learning Engineer": """
Job Title: Machine Learning Engineer

Responsibilities:
- Build and deploy ML models using real-world data
- Preprocess and clean data for training and evaluation
- Work with data scientists and engineers to build scalable systems
- Continuously improve model performance

Required Skills:
- Strong Python programming (Pandas, NumPy, Scikit-learn)
- Knowledge of ML algorithms (classification, regression, clustering)
- Experience with TensorFlow or PyTorch
- Familiarity with evaluation metrics (precision, recall, F1-score)

Bonus:
- Knowledge of MLOps tools like MLFlow, Airflow
- Experience deploying models to cloud or edge devices
""",

    "Backend Developer": """
Job Title: Backend Developer

Responsibilities:
- Design and build scalable backend systems and REST APIs
- Manage and maintain databases
- Optimize performance and implement caching
- Ensure security and handle authentication/authorization

Required Skills:
- Proficiency in Python (Flask/Django), Node.js, or Java
- Experience with SQL and NoSQL databases
- Understanding of RESTful API design
- Familiarity with Git, Postman, and backend debugging

Bonus:
- Docker, Kubernetes, or CI/CD knowledge
- Cloud experience with AWS, GCP, or Azure
""",

    "Frontend Developer": """
Job Title: Frontend Developer

Responsibilities:
- Convert design mockups into interactive web interfaces
- Optimize websites for performance and accessibility
- Ensure cross-browser compatibility
- Implement UI components using frontend frameworks

Required Skills:
- HTML, CSS, JavaScript (ES6+)
- Experience with React, Vue, or Angular
- Knowledge of responsive and mobile-first design
- Familiarity with Git and browser dev tools

Bonus:
- Experience with Tailwind, Bootstrap, or Material UI
- Basic knowledge of REST APIs
""",

    "UI/UX Designer": """
Job Title: UI/UX Designer

Responsibilities:
- Conduct user research and create personas
- Design wireframes, prototypes, and user flows
- Collaborate with developers and product managers
- Ensure consistency and usability across interfaces

Required Skills:
- Tools like Figma, Adobe XD, Sketch
- Understanding of UX principles and usability testing
- Design thinking and problem-solving mindset
- Good communication and attention to detail

Bonus:
- Basic HTML/CSS skills
- Experience working in Agile/Scrum environments
""",

    "Cloud Engineer": """
Job Title: Cloud Engineer

Responsibilities:
- Manage and monitor cloud infrastructure (AWS, GCP, Azure)
- Automate provisioning and scaling using IaC tools
- Ensure availability, performance, and security of systems
- Support cloud-based application deployment

Required Skills:
- Experience with AWS, Azure, or GCP services
- Knowledge of networking, VPC, load balancing, and firewalls
- Familiarity with Terraform, CloudFormation, or Ansible
- Monitoring tools like CloudWatch, Grafana

Bonus:
- Certification (AWS Certified Solutions Architect, etc.)
- Experience with Kubernetes or serverless architectures
""",

    "DevOps Engineer": """
Job Title: DevOps Engineer

Responsibilities:
- Implement CI/CD pipelines and release management
- Automate infrastructure using scripts and configuration tools
- Ensure application monitoring and incident response
- Collaborate with developers and QA for seamless releases

Required Skills:
- Experience with Jenkins, GitHub Actions, or GitLab CI
- Proficiency in Docker, Kubernetes
- Scripting in Bash, Python, or Shell
- Familiarity with Linux environments and logging tools

Bonus:
- Cloud infrastructure knowledge (AWS, GCP)
- Monitoring tools (Prometheus, ELK Stack, Datadog)
""",

    "Product Manager": """
Job Title: Product Manager

Responsibilities:
- Define product vision and roadmap
- Collect user requirements and prioritize features
- Collaborate with engineering and design teams
- Track product metrics and iterate based on feedback

Required Skills:
- Strong communication and stakeholder management
- Experience with tools like JIRA, Trello, Notion
- Analytical skills using Excel, SQL, or Google Analytics
- Agile/Scrum methodology understanding

Bonus:
- Technical background (CS/IT or engineering)
- MBA or business exposure
"""
}

# List of job roles for UI dropdown (for both recruiter and job seeker)
all_available_job_roles = list(job_roles.keys())
# Add "Custom Job Role" to the job seeker options for flexible input.
job_seeker_role_names = all_available_job_roles + ["Custom Job Role"]
# Recruiter also gets "Custom Job Role" option for job descriptions.
recruiter_role_options = all_available_job_roles + ["Custom Job Role"]


# --- Initialize and Fit TF-IDF Vectorizer for Job Seeker Mode ---
# This TF-IDF vectorizer needs to be fitted once on a representative corpus
# for the Job Seeker's cosine similarity calculations.
# We'll use all the predefined job descriptions to build its vocabulary.
job_description_texts = list(job_roles.values())
cleaned_job_description_texts = [clean_text(jd) for jd in job_description_texts]

tfidf_job_seeker = TfidfVectorizer()
tfidf_job_seeker.fit(cleaned_job_description_texts)

print("✅ Job roles loaded and Job Seeker TF-IDF vectorizer initialized and fitted.")


# --- Recruiter Model's Core Logic ---

def recruiter_find_matches(valid_resume_contents_with_names, job_description_text, top_n_slider):
    """
    Finds and ranks resumes based on their similarity to a given job description,
    designed for the recruiter's perspective.

    Args:
        valid_resume_contents_with_names (list of tuples): A list where each tuple contains
                                                            (resume_text_content, original_file_name)
                                                            for resumes that passed initial validation.
        job_description_text (str): The text of the job description.
        top_n_slider (int): The number of top matching resumes to display in the UI.

    Returns:
        tuple: A tuple containing:
            - str: Formatted text output for Gradio UI displaying top matches and processing summary.
            - str: File path to a temporary CSV report containing detailed match results, or "" if no valid resumes.
    """
    # Separate contents and names for valid resumes
    resumes_clean = [clean_text(r[0]) for r in valid_resume_contents_with_names]
    original_valid_resume_names = [r[1] for r in valid_resume_contents_with_names] # Extract original file names

    job_description_clean = clean_text(job_description_text)

    # Handle cases where no valid resumes are provided
    if not resumes_clean:
        return "No valid resumes to process for matching.", "" # Return empty string for CSV path

    # Combine job description and all cleaned resumes for TF-IDF vectorization
    all_documents = [job_description_clean] + resumes_clean

    # Initialize TF-IDF Vectorizer specific to the recruiter's batch processing
    vectorizer_recruiter = TfidfVectorizer(
        stop_words='english',
        min_df=1,       # Consider terms appearing in at least 1 document
        max_df=0.99,    # Ignore terms appearing in almost all documents (too common)
        strip_accents='unicode' # Remove accent marks from characters
    )

    # Fit and transform all documents into a TF-IDF matrix
    tfidf_matrix = vectorizer_recruiter.fit_transform(all_documents)

    # Check if there's enough relevant text to perform similarity calculation
    # (i.e., at least the JD and one valid resume with content)
    if tfidf_matrix.shape[0] < 2:
        return "Not enough relevant text for matching after cleaning. Ensure JD and valid resumes have sufficient content.", ""

    # Calculate cosine similarity of the Job Description (first row, index 0)
    # with all valid resumes (subsequent rows, starting from index 1)
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    sorted_indices = np.argsort(scores)[::-1] # Sort indices in descending order of similarity score

    raw_results = []
    # Define the match threshold for flagging resumes as 'MATCH' or 'NO MATCH'
    match_threshold = 15 # Percentage threshold (e.g., 15% for TF-IDF scores)

    # Iterate through sorted results to compile raw data
    for idx, i in enumerate(sorted_indices):
        score = round(float(scores[i]) * 100, 2) # Convert similarity score to percentage
        is_match = score >= match_threshold

        raw_results.append({
            "Rank": idx + 1,
            "Original Resume Name": original_valid_resume_names[i], # Use the actual file name
            "Score (%)": score,
            "is_match_flag": is_match,
            # "Resume Text": valid_resume_contents_with_names[i][0] # Optional: keep for internal debug/full report
        })

    # Prepare results for Gradio UI display (with emojis and truncated view based on top_n_slider)
    top_resumes_for_display = []
    if raw_results:
        for item in raw_results[:top_n_slider]:
            display_label = ""
            if item["is_match_flag"]:
                display_label = f"✅ MATCH - {item['Score (%)']}%"
            else:
                display_label = f"❌ NO MATCH (Score: {item['Score (%)']}%)"
            top_resumes_for_display.append(f"🔹 Rank {item['Rank']} - {display_label}\n    Resume Name: {item['Original Resume Name']}")
    else:
        top_resumes_for_display.append("No matches found among valid resumes for the given job description and threshold.")

    # Prepare df_report (Pandas DataFrame) for CSV download
    df_report = pd.DataFrame(raw_results)
    df_report['Match Type'] = df_report.apply(
        lambda row: f"MATCH - {row['Score (%)']}%" if row['is_match_flag']
                            else f"NO MATCH (Score: {row['Score (%)']}%)",
        axis=1
    )
    # Remove the internal 'is_match_flag' column as it's represented by 'Match Type'
    df_report = df_report.drop(columns=['is_match_flag'], errors='ignore')
    # If "Resume Text" was added for internal debugging, ensure it's dropped for the final CSV
    if "Resume Text" in df_report.columns:
        df_report = df_report.drop(columns=['Resume Text'], errors='ignore')

    # Create a temporary CSV file for download, ensuring it's not deleted immediately
    # The file path is returned to Gradio, which then handles the download.
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False, encoding="utf-8") as tmp_file:
        df_report.to_csv(tmp_file.name, index=False)
        csv_file_path = tmp_file.name

    # Join display parts with double newlines for better readability in Gradio Textbox
    return "\n\n".join(top_resumes_for_display), csv_file_path

print("✅ Recruiter matching logic defined.")


# --- Job Seeker Model's Core Logic ---

def job_seeker_calculate_match(resume_text, input_job_description_text):
    """
    Compares a given resume text to a given job description text using the pre-fitted TF-IDF
    vectorizer (tfidf_job_seeker) and Cosine Similarity.

    Args:
        resume_text (str): The cleaned text content of the job seeker's resume.
        input_job_description_text (str): The cleaned text content of the job description.

    Returns:
        str: A formatted prediction string indicating 'MATCH' or 'NO MATCH'
             along with the similarity score and the threshold for context.
    """
    # Define the percentage threshold for a 'MATCH' in Job Seeker mode
    # For example, 0.45 means 45% similarity or higher is considered a match.
    match_threshold = 0.45

    # Clean both the resume and job description texts
    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(input_job_description_text)

    # Transform the cleaned texts into TF-IDF vectors using the globally fitted tfidf_job_seeker.
    # This ensures consistency with the vocabulary built from predefined job roles.
    resume_tfidf = tfidf_job_seeker.transform([cleaned_resume])
    jd_tfidf = tfidf_job_seeker.transform([cleaned_jd])

    # Calculate cosine similarity between the resume and job description TF-IDF vectors.
    # The result is a 2D array, so [0][0] extracts the single scalar similarity score.
    similarity_score = cosine_similarity(resume_tfidf, jd_tfidf)[0][0]
    score_percentage = round(float(similarity_score) * 100, 2) # Convert to a percentage

    # Determine if it's a 'MATCH' based on the defined threshold and return the formatted string.
    if score_percentage >= match_threshold * 100:  # Convert threshold to percentage for comparison
        return f"✅ MATCH! Your Resume Score: {score_percentage:.2f}% (Threshold: {match_threshold*100:.0f}%)"
    else:
        return f"❌ NO MATCH! Your Resume Score: {score_percentage:.2f}% (Threshold: {match_threshold*100:.0f}%)"

print("✅ Job Seeker matching logic defined.")


# --- Gradio Helper Functions - FINAL UPDATE (Soft Warning for Job Seeker Custom JD) ---

# Make sure gradio is imported for gr.Info (used for non-blocking messages) - already imported at top

def process_recruiter_request(selected_recruiter_role, custom_jd_text, resume_file_paths_list, top_n_slider):
    """
    Processes the recruiter's request to match multiple resumes against a job description.

    Args:
        selected_recruiter_role (str): The job role selected from the dropdown,
                                        or "Custom Job Role".
        custom_jd_text (str): Custom job description text provided by the recruiter.
                                Used if selected_recruiter_role is "Custom Job Role".
        resume_file_paths_list (list): A list of file paths for uploaded resumes.
        top_n_slider (int): The number of top matching resumes to display in the UI.

    Returns:
        tuple: A tuple containing:
            - str: Formatted text output for Gradio UI displaying processing summary and top matches.
            - str: File path to a temporary CSV report containing detailed match results, or "" for errors/no valid resumes.
    """
    # Determine the Job Description source
    jd_content = ""
    if selected_recruiter_role == "Custom Job Role":
        if not custom_jd_text.strip():
            # Return empty string for CSV path for error cases
            return "❌ Please provide a custom Job Description text when 'Custom Job Role' is selected.", ""
        jd_content = custom_jd_text
    else:
        # Fetch predefined JD if a specific role is selected
        jd_content = job_roles.get(selected_recruiter_role, "")
        if not jd_content.strip():
            return f"❌ Job Description for role '{selected_recruiter_role}' not found. Please select another role.", ""

    # Validate resume uploads
    if not resume_file_paths_list:
        return "❌ Please upload at least one Resume.", ""

    total_resumes_uploaded = len(resume_file_paths_list)
    valid_resume_contents_with_names = []
    invalid_resumes_info = [] # To store (name, reason) for invalid resumes

    # Process each uploaded resume file
    for r_file_path in resume_file_paths_list:
        file_name = os.path.basename(r_file_path)
        r_content = extract_text_from_file(r_file_path)

        if r_content.startswith("❌"): # Check for file extraction errors returned by extract_text_from_file
            invalid_resumes_info.append((file_name, r_content)) # Store error message
        elif not r_content.strip(): # Check if content is empty after extraction
            invalid_resumes_info.append((file_name, "Empty or unreadable content."))
        elif not is_probably_resume(r_content): # Use the improved heuristic to validate resume content
            invalid_resumes_info.append((file_name, "Content does not resemble a resume (e.g., missing key sections or contains non-resume indicators)."))
        else:
            valid_resume_contents_with_names.append((r_content, file_name))

    num_valid_resumes = len(valid_resume_contents_with_names)
    num_invalid_resumes = len(invalid_resumes_info)

    # Building the display text for the UI to summarize processing
    display_text_parts = [
        f"--- Resume Processing Summary ---",
        f"Total Resumes Uploaded: {total_resumes_uploaded}",
        f"Valid Resumes Processed for Matching: {num_valid_resumes}",
        f"Invalid/Problematic Resumes Detected: {num_invalid_resumes}"
    ]

    # Add details for invalid resumes if any
    if num_invalid_resumes > 0:
        display_text_parts.append("\n--- Invalid Resumes ---")
        for name, reason in invalid_resumes_info:
            display_text_parts.append(f"❌ {name}: {reason}")

    display_text_parts.append("\n--- Top Matching Valid Resumes ---")

    # If no valid resumes remain after filtering, inform the user
    if num_valid_resumes == 0:
        display_text_parts.append("No valid resumes to perform matching against the Job Description.")
        return "\n".join(display_text_parts), "" # Return empty CSV path if no valid resumes

    try:
        # Call the core recruiter_find_matches function with only valid resumes
        top_matches_display_string, csv_file_path = recruiter_find_matches(
            valid_resume_contents_with_names, jd_content, top_n_slider
        )

        display_text_parts.append(top_matches_display_string)

        return "\n".join(display_text_parts), csv_file_path

    except Exception as e:
        # Catch any unexpected errors during the matching process
        return f"An unexpected error occurred during matching: {str(e)}", ""


def process_job_seeker_request(selected_role, resume_text_input, resume_file_path, custom_jd_text_js):
    """
    Processes the job seeker's request to match their resume against a job description.

    Args:
        selected_role (str): The job role selected from the dropdown,
                                or "Custom Job Role".
        resume_text_input (str): Resume text directly pasted by the user (optional).
        resume_file_path (str): Path to an uploaded resume file (optional).
        custom_jd_text_js (str): Custom job description text provided by the job seeker (optional).

    Returns:
        tuple: A tuple containing:
            - str: Formatted text output for Gradio UI displaying the match score.
            - None: Always None for job seeker mode, as no CSV is generated.
    """
    resume_content = ""
    # Prioritize file upload if provided and valid
    if resume_file_path is not None and isinstance(resume_file_path, str) and os.path.exists(resume_file_path):
        resume_content = extract_text_from_file(resume_file_path)
        if resume_content.startswith("❌"): # Check for file extraction errors
            return resume_content, None
        elif not resume_content.strip():
            return "❌ Uploaded file is empty or unreadable.", None
    elif resume_text_input and resume_text_input.strip(): # Fallback to text input if no valid file
        resume_content = resume_text_input
    else: # Neither text nor valid file path provided
        return "❌ Please enter your resume text or upload a resume file.", None

    if not resume_content.strip():
        return "❌ Please enter or upload a resume that contains text.", None

    # Validate if it's likely a resume using the improved heuristic function
    if not is_probably_resume(resume_content):
        return "❌ This doesn't look like a valid resume. Please ensure it contains sections like 'Experience', 'Skills', 'Education', etc., or has sufficient text content.", None

    jd_content = ""
    # Core Logic for Job Seeker JD selection:
    # 1. Priority: Use custom JD if provided in the dedicated custom_jd_text_js box.
    if custom_jd_text_js and custom_jd_text_js.strip(): # Corrected from custom_jd_js to custom_jd_text_js
        jd_content = custom_jd_text_js.strip()

        # NEW: Soft warning for short custom JDs to encourage comprehensive input
        if len(jd_content) < 100: # Example: warn if less than 100 characters
            gr.Info("💡 Heads Up! The custom Job Description you entered seems very short. For best results, please paste the complete job description from the job posting.")

    # 2. If 'Custom Job Role' is selected but no custom JD text was provided.
    elif selected_role == "Custom Job Role":
        return "❌ When 'Custom Job Role' is selected, you must provide a Job Description.", None
    # 3. Last Resort: Use predefined JD for the selected role from the `job_roles` dictionary.
    elif selected_role:
        jd_content = job_roles.get(selected_role, "")

    # Final check to ensure a Job Description is available for matching
    if not jd_content.strip():
        return f"❌ A Job Description is required for matching. Please select a role or provide a custom JD.", None

    try:
        # Call the core job_seeker_calculate_match function to get the match result
        match_result = job_seeker_calculate_match(resume_content, jd_content)
        return match_result, None # Return None for the file output, as job seeker doesn't generate a CSV

    except Exception as e:
        # Catch any unexpected errors during the matching process
        return f"An unexpected error occurred in Job Seeker mode: {str(e)}", None

print("✅ Gradio helper functions updated with Job Seeker custom JD soft warning.")


# --- Combined Gradio Interface Definition and Launch (Updated Job Seeker Custom JD Label) ---

# Ensure job_roles is accessible (assuming it's defined in a previous cell).
# This block provides a placeholder definition for robustness in case of partial runs,
# preventing errors if job_roles hasn't been executed.
if 'job_roles' not in globals():
    job_roles = {
        "Data Scientist": "A data scientist analyzes complex data to extract insights and inform decision-making. Requires strong skills in statistics, machine learning, Python/R, SQL, and data visualization.",
        "Software Engineer": "A software engineer designs, develops, and maintains software applications. Requires proficiency in programming languages (e.g., Python, Java, C++), data structures, algorithms, and software development methodologies.",
        "Product Manager": "A product manager defines the product vision, strategy, and roadmap. Requires strong communication, market analysis, technical understanding, and leadership skills.",
        "UX/UI Designer": "A UX/UI designer focuses on user experience and interface design, creating intuitive and aesthetically pleasing digital products. Requires skills in user research, wireframing, prototyping, and design tools.",
        "Marketing Specialist": "A marketing specialist develops and implements marketing campaigns, analyzes market trends, and manages brand presence. Requires skills in digital marketing, content creation, SEO/SEM, and analytics.",
        "HR Manager": "An HR manager oversees human resources functions, including recruitment, employee relations, and compliance."
    }

# Ensure job role options lists are correctly generated from job_roles.
# This ensures the dropdowns are populated correctly even if preceding parts were not run in order.
if 'all_available_job_roles' not in globals():
    all_available_job_roles = list(job_roles.keys())
if 'recruiter_role_options' not in globals():
    recruiter_role_options = all_available_job_roles + ["Custom Job Role"]
if 'job_seeker_role_names' not in globals():
    job_seeker_role_names = all_available_job_roles + ["Custom Job Role"]


with gr.Blocks(title="Unified Resume Matcher App") as demo:
    gr.Markdown("# Welcome to the Unified Resume Matching Application!")
    gr.Markdown("## Select Your Role:")

    # Initial Role Selection Radio buttons to switch between Recruiter and Job Seeker UI
    role_selector = gr.Radio(
        ["Recruiter", "Job Seeker"],
        label="Are you a Recruiter or a Job Seeker?",
        value="Recruiter" # Default selected role when the app loads
    )

    # --- Recruiter Mode UI Group ---
    # This group contains all UI elements specific to the recruiter's functionality.
    # It is initially visible by default.
    with gr.Group(visible=True) as recruiter_group:
        gr.Markdown("### Recruiter Dashboard: Match multiple resumes to a Job Description")
        with gr.Row():
            recruiter_role_dropdown = gr.Dropdown(
                recruiter_role_options, # Populated with predefined roles and "Custom Job Role"
                label="Select Job Role or Custom",
                value=recruiter_role_options[0], # Set a default selected value
                allow_custom_value=False # Users must select from the list
            )
            # This textbox for custom JD will be conditionally visible based on dropdown selection
            recruiter_custom_jd_text = gr.Textbox(
                label="Enter Custom Job Description",
                interactive=True,
                lines=5,
                placeholder="Paste your custom Job Description here...",
                visible=False # Initially hidden
            )

        # Logic to show/hide the custom JD textbox based on the recruiter role dropdown selection.
        # When "Custom Job Role" is selected, the textbox becomes visible.
        recruiter_role_dropdown.change(
            fn=lambda role: gr.Textbox(visible=(role == "Custom Job Role")),
            inputs=recruiter_role_dropdown,
            outputs=recruiter_custom_jd_text
        )

        recruiter_resume_files = gr.File(
            label="Upload Resumes (PDF/DOCX, multiple)",
            file_count="multiple", # Allows multiple file uploads
            type="filepath",        # Returns the file path for processing
            file_types=[".pdf", ".docx"] # Specifies allowed file extensions
        )

        recruiter_top_n_input = gr.Slider(
            minimum=1,
            maximum=20,
            value=5, # Default value
            step=1,
            label="Display Top N Resumes"
        )
        recruiter_match_button = gr.Button("Find Matches for Recruiter")

        recruiter_output_text = gr.Textbox(
            label="Recruiter Match Results",
            interactive=False, # User cannot type in this box
            lines=20 # Increased lines for more comprehensive output display
        )
        recruiter_download_csv = gr.File(
            label="Download Full Match Report (CSV)",
            interactive=False, # File component acts as a download link
            file_count="single",
            type="filepath"
        )

        # Define the action for the recruiter match button click
        recruiter_match_button.click(
            fn=process_recruiter_request, # Calls the backend processing function
            inputs=[
                recruiter_role_dropdown,
                recruiter_custom_jd_text,
                recruiter_resume_files,
                recruiter_top_n_input
            ],
            outputs=[recruiter_output_text, recruiter_download_csv]
        )

    # --- Job Seeker Mode UI Group ---
    # This group contains all UI elements specific to the job seeker's functionality.
    # It is initially hidden and becomes visible when 'Job Seeker' is selected.
    with gr.Group(visible=False) as job_seeker_group:
        gr.Markdown("### Job Seeker Tool: Check your resume against a specific role")
        job_seeker_role_dropdown = gr.Dropdown(
            job_seeker_role_names, # Populated with predefined roles and "Custom Job Role"
            label="Select Your Desired Job Role or 'Custom Job Role'",
            value=job_seeker_role_names[0] if job_seeker_role_names else None, # Set a default value if available
            allow_custom_value=False
        )
        # Custom JD input for Job Seeker. It starts visible but its label/placeholder changes.
        job_seeker_custom_jd_input = gr.Textbox(
            label="Optional: Paste Your Custom Job Description here (overrides the predefined JD)", # ✅ UPDATED LABEL
            placeholder="Paste the specific Job Description you are applying for...",
            lines=5,
            interactive=True,
            visible=True # This makes it visible by default when the tab is active
        )

        with gr.Row(): # Group resume input options in a row
            job_seeker_resume_text = gr.Textbox(
                lines=8,
                label="Resume Text (optional)",
                placeholder="Paste resume text if not uploading"
            )
            job_seeker_resume_file = gr.File(
                label="Upload Your Resume (.pdf or .docx)",
                file_types=[".pdf", ".docx"],
                type="filepath"
            )

        job_seeker_match_button = gr.Button("Check My Resume Match")

        job_seeker_output_text = gr.Textbox(
            label="Your Match Score",
            interactive=False
        )
        # Dummy file output for consistent function signature with process_recruiter_request
        job_seeker_dummy_file_output = gr.File(visible=False)

        # Logic to dynamically update the label/placeholder of the custom JD textbox
        # for the Job Seeker based on dropdown selection.
        def update_job_seeker_jd_display(selected_role_js):
            if selected_role_js == "Custom Job Role":
                return gr.Textbox(
                    label="Please Paste Your Custom Job Description Here (Mandatory for 'Custom Job Role')",
                    placeholder="Enter the full custom Job Description here...",
                    value="" # Clear previous value to encourage new input
                )
            elif selected_role_js: # Any other predefined role selected (e.g., Data Analyst, SWE)
                return gr.Textbox(
                    label="Optional: Paste Your Custom Job Description here (overrides the predefined JD)", # ✅ UPDATED LABEL
                    placeholder="Paste the specific Job Description you are applying for...",
                    value="" # Clear previous value
                )
            else: # If dropdown is cleared (should rarely happen with default value)
                return gr.Textbox(
                    label="Optional: Paste Your Custom Job Description here (overrides the predefined JD)", # Default label
                    placeholder="",
                    value="" # Clear value
                )

        job_seeker_role_dropdown.change(
            fn=update_job_seeker_jd_display,
            inputs=job_seeker_role_dropdown,
            outputs=job_seeker_custom_jd_input
        )

        # Define the action for the job seeker match button click
        job_seeker_match_button.click(
            fn=process_job_seeker_request, # Calls the backend processing function
            inputs=[
                job_seeker_role_dropdown,
                job_seeker_resume_text,
                job_seeker_resume_file,
                job_seeker_custom_jd_input # Pass the custom JD input
            ],
            outputs=[job_seeker_output_text, job_seeker_dummy_file_output]
        )

    # --- Logic to toggle group visibility based on the main role selection ---
    # This function determines which UI group (Recruiter or Job Seeker) is visible.
    def set_visibility(choice):
        if choice == "Recruiter":
            return gr.Group(visible=True), gr.Group(visible=False)
        else: # Job Seeker
            return gr.Group(visible=False), gr.Group(visible=True)

    role_selector.change(
        fn=set_visibility,
        inputs=role_selector,
        outputs=[recruiter_group, job_seeker_group]
    )

# Launch the Gradio application
demo.launch(debug=True, share=True)