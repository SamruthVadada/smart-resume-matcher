---
title: Smart Resume Matcher
emoji: 🧠📄
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.36.1"
app_file: app.py # <--- CRITICAL: Points to app.py
pinned: false
---

# Smart Resume Matcher – AI Tool for Job Seekers & Recruiters 🧠📄

An AI-powered resume matching tool built using **Gradio**, **Python**, and **NLP** techniques, designed to help **recruiters** find top candidates and assist **job seekers** in evaluating their resume-fit for a job.

---

## 🚀 Features

🔹 **Recruiter Mode**
Upload multiple resumes and match them against a job description to get a ranked list of top candidates.
Export detailed match scores as a downloadable CSV.

🔹 **Job Seeker Mode**
Upload or paste your resume, enter a job description, and instantly receive a personalized match score.

🔹 **Custom + Predefined Job Descriptions**
Use your own JD or select from built-in roles like *Data Scientist*, *Software Engineer*, *UX Designer*, etc.

🔹 **Intelligent Resume Validation**
Detects and filters out non-resume files (like certificates or blank documents).

🔹 **Powerful Text Matching**
Uses **TF-IDF** and **Cosine Similarity** for accurate content comparison.

🔹 **User-Friendly Interface**
Built with **Gradio** for a clean, interactive experience.

---

## 📁 Project Structure

📦 Smart Resume Matcher/
├── app.py                  # Main Python App File
├── requirements.txt        # All Python dependencies
└── README.md               # Project overview and documentation

---

## 🛠️ Technologies Used

- Python
- Gradio
- Scikit-learn
- PyMuPDF (fitz)
- python-docx
- pandas & numpy
- re, os, tempfile
- spacy (model downloaded at runtime)

---

## ✅ How to Run

### 💻 Option 1: Run Locally

To set up and run the project on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone [YOUR_NEW_GITHUB_REPO_URL_HERE] # Replace with your new GitHub repo URL
    cd smart-resume-matcher
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch the Gradio app:**
    ```bash
    python app.py
    ```
    This command will launch the Gradio interface in your default web browser, typically at `http://127.0.0.1:7860`.

---

## 🌐 Live Demo

Experience the Smart Resume Matcher in action right now on Hugging Face Spaces!

🔗 **Access the Live Application Here:** [YOUR_NEW_HUGGING_FACE_SPACE_URL_HERE] # This will be the URL of your new Space

---

## 🙋‍♂️ Developer

**Samruth Vadada**
* **Email:** vadadasamruth@gmail.com
* **GitHub:** [https://github.com/SamruthVadada/smart-resume-matcher](https://github.com/SamruthVadada/smart-resume-matcher)

---

## 📄 License

This project is open-source and distributed under the **MIT License**. See the `LICENSE` file in the repository for full details.