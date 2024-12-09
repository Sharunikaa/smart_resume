import spacy
import PyPDF2
import nltk
from flask import Flask, request, render_template, session
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from bs4 import BeautifulSoup
from docx import Document  # Import Document for .docx file handling
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk import CFG
from nltk.tree import Tree
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import scrolledtext
import threading
from nltk.tokenize import word_tokenize
import logging

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'secret_key'  # Set a secret key for session management

# Initialize spaCy model
nlp = spacy.load("en_core_web_md")

data = {
    "phrase": [
        # Skills
        "Python", "Data Science", "Machine Learning", "SQL", "Java", "Business Intelligence",
        "Statistical Analysis", "Data Visualization", "Tableau", "TensorFlow", "NLP",
        "Deep Learning", "Hadoop", "Spark", "Data Engineer", "Data Mining", "R", "Excel",
        "Power BI", "SAS", "Django", "Flask", "JavaScript", "HTML", "CSS", "C++",
        "Ruby on Rails", "Cloud Computing", "AWS", "Azure", "Kubernetes", "Docker",
        "Agile Methodologies", "Scrum", "JIRA", "Git", "REST APIs", "Microservices",
        "Digital Marketing", "Content Strategy", "Social Media Management", "SEO",
        "UX Design", "UI Design", "Adobe Photoshop", "Adobe Illustrator", "Graphics Design",
        "Cybersecurity", "Information Security", "Network Administration", "ITIL",
        "Salesforce", "CRM Software", "Excel VBA", "Data Governance", "Data Quality Management",
        "Statistical Software", "MATLAB", "MongoDB", "PostgreSQL", "MySQL", "NoSQL",
        "Big Data Analytics", "Artificial Intelligence", "Natural Language Processing",
        "Computer Vision", "Data Warehousing", "ETL", "Data Architecture", "Data Cataloging",
        # Non-Skills
        "Project Management", "Communication", "Teamwork", "Leadership",
        "Bachelor's Degree", "3 years of experience", "Full-time", "Remote",
        "Worked in various teams", "Strong interpersonal skills", "Attended workshops",
        "Participated in hackathons", "Graduated with honors", "Trained new employees",
        "Worked in a fast-paced environment", "Adaptability", "Customer service experience",
        "Fluent in English and Spanish", "Certified Scrum Master", "Mentored interns",
        "Conducted presentations", "Wrote technical documentation", "Managed budgets",
        "Oversaw project timelines", "Performed market research", "Developed training materials",
        "Familiar with Microsoft Office", "Worked on multiple projects simultaneously",
        "Conducted user interviews", "Facilitated team meetings", "Participated in code reviews",
        "Created user manuals", "Assisted in recruitment", "Maintained a positive work environment",
        "Worked on customer feedback", "Ensured compliance with regulations"
    ]
}

# Define a set of known skills to cross-reference with extracted skills
known_skills = set(data["phrase"])
# Sample general interview questions and answers
general_interview_questions = [
    {
        "question": "What is your greatest strength?",
        "expected_answer": "My greatest strength is my ability to solve complex problems efficiently."
    },
    {
        "question": "Describe a challenge you faced at work and how you dealt with it.",
        "expected_answer": "I faced a significant project deadline and organized my team to prioritize tasks and meet the deadline."
    },
    {
        "question": "Why do you want to work for this company?",
        "expected_answer": "I admire this company’s commitment to innovation and believe my skills align well with your goals."
    }
]

# Sample job-specific interview questions
job_specific_questions = {
    "Software Developer": [
        {
            "question": "Can you explain the difference between a list and a tuple in Python?",
            "expected_answer": "Yes, a list is mutable, while a tuple is immutable."
        },
        {
            "question": "How do you manage memory in Java?",
            "expected_answer": "Java uses automatic garbage collection to manage memory."
        }
    ],
    "Data Analyst": [
        {
            "question": "What is the purpose of data normalization?",
            "expected_answer": "Data normalization is used to minimize redundancy and dependency by organizing fields."
        },
        {
            "question": "Can you explain what a pivot table is?",
            "expected_answer": "A pivot table is a data processing tool used to summarize data from a larger table."
        }
    ],
    "Big Data Engineer": [
        {
            "question": "What are the differences between Hadoop and Spark?",
            "expected_answer": "Hadoop is a batch processing framework, while Spark provides in-memory data processing."
        },
        {
            "question": "Can you explain what a data lake is?",
            "expected_answer": "A data lake is a centralized repository that stores all structured and unstructured data."
        }
    ]
}

# Text Preprocessing
def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text)  # Remove non-alphanumeric characters
    text = text.lower()  # Lowercase the text
    return text
def cfg_parser(input_text):
    # Tokenize the sentence into words
    tokens = word_tokenize(input_text)

    # POS tagging each word in the sentence
    tagged_words = pos_tag(tokens)

    # Automatically generate grammar rules for common parts of speech
    grammar = CFG.fromstring("""
      S -> NP VP
      NP -> DT NN | DT JJ NN | PRP | NN
      VP -> VBZ NP | VBD NP | VBP NP | VB VP | VBD PP | VBG PP | VBZ PP | VBN PP
      PP -> IN NP
      DT -> 'the' | 'The'
      NN -> 'dog' | 'fox' | 'cat' | 'apple' | 'ball' | 'man'
      PRP -> 'he' | 'she' | 'it' | 'I' | 'you' | 'we'
      VBZ -> 'jumps' | 'chases' | 'runs'
      VBD -> 'chased' | 'ran' | 'ate'
      VBP -> 'run'
      VBG -> 'running'
      VBN -> 'eaten'
      VB -> 'chase' | 'run' | 'eat'
      IN -> 'over' | 'under' | 'on' | 'above'
      JJ -> 'quick' | 'brown' | 'lazy' | 'happy' | 'sad'
    """)

    # Parse the sentence using the defined grammar
    parser = nltk.ChartParser(grammar)

    # Try to parse the sentence based on the given grammar
    try:
        parsed_sentence = list(parser.parse(tokens))
        return tagged_words, parsed_sentence  # Return tagged words and parse trees
    except ValueError as e:
        print(f"\nError: {e}")
        return tagged_words, []  # Return empty list if parsing fails

def display_parse_tree(parse_trees):
    window = tk.Tk()
    window.title("Parse Tree Display")

    text_area = scrolledtext.ScrolledText(window, width=80, height=20)
    text_area.pack(padx=10, pady=10)

    for tree in parse_trees:
        text_area.insert(tk.END, tree.pformat() + "\n\n")
        # Optionally, visualize the tree
        tree.draw()  # Open a new window for tree visualization

    text_area.config(state=tk.DISABLED)
    window.mainloop()

def run_tkinter(parse_trees):
    display_parse_tree(parse_trees)

def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(docx_file):
    text = ""
    doc = Document(docx_file)
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_skills(resume_text):
    resume_text = resume_text.lower()
    skills_found = set()

    # Use NER to find potential skills in the text
    doc = nlp(resume_text)

    # Check for known skills in NER entities
    for entity in doc.ents:
        if entity.label_ in ['PRODUCT', 'SKILL', 'ORG']:
            skill = entity.text.lower()
            if skill in known_skills:
                skills_found.add(skill)

    # Check for known skills in token matches and noun chunks
    for token in doc:
        skill = token.text.lower()
        if skill in known_skills:
            skills_found.add(skill)

    # Check for multi-word skills
    for chunk in doc.noun_chunks:
        multi_word_skill = ' '.join([token.text for token in chunk]).lower()
        if multi_word_skill in known_skills:
            skills_found.add(multi_word_skill)

    # Include a check for exact skills from known_skills
    for known_skill in known_skills:
        if known_skill.lower() in resume_text:
            skills_found.add(known_skill.lower())

    return list(skills_found)

def calculate_semantic_similarity(resume_text, job_description):
    resume_doc = nlp(resume_text)
    job_doc = nlp(job_description)
    return resume_doc.similarity(job_doc) * 100

# Function to score the resume
def score_resume(resume_text, job_description):
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)
    semantic_similarity = calculate_semantic_similarity(resume_text, job_description)

    # Calculate skill match percentage
    matching_skills = len(set(resume_skills) & set(job_skills))
    skill_match_percentage = (matching_skills / len(set(job_skills))) * 100 if job_skills else 0

    # Ensure there’s a score even if no job skills are matched
    if not job_skills and resume_skills:
        skill_match_percentage = 100  # All skills in the resume are valid

    # TF-IDF based cosine similarity score
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])

    # Correctly get the cosine similarity value
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    cosine_score = cosine_sim[0][0] * 100  # Ensure this is a scalar value

    # Adjusted total score with increased weight for skill match
    total_score = (0.4 * skill_match_percentage) + (0.3 * semantic_similarity) + (0.3 * cosine_score)

    return {
        "total_score": round(total_score, 2),
        "skill_match_percentage": round(skill_match_percentage, 2),
        "semantic_similarity": round(semantic_similarity, 2),
        "cosine_similarity": round(cosine_score, 2),
        "skills_found": resume_skills
    }

def classify_job_category(resume_text):
    job_descriptions = [
        {"description": "Looking for a software developer proficient in Python, Java, and SQL",
         "category": "Software Developer"},
        {"description": "Data analyst with experience in machine learning, statistics, and TensorFlow",
         "category": "Data Analyst"},
        {"description": "Big data engineer with knowledge of Hadoop, Spark, and Power BI",
         "category": "Big Data Engineer"}
    ]

    X = [job['description'] for job in job_descriptions]
    y = [job['category'] for job in job_descriptions]

    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)
    resume_vectorized = vectorizer.transform([resume_text])

    cosine_similarities = cosine_similarity(resume_vectorized, X_vectorized).flatten()
    return y[cosine_similarities.argmax()]
# Function to compare answers
def compare_answers(user_answer, expected_answer):
    # Normalize both answers to lower case and strip whitespace for comparison
    normalized_user_answer = user_answer.strip().lower()
    normalized_expected_answer = expected_answer.strip().lower()

    # Debugging output
    print(f"Normalized User Answer: '{normalized_user_answer}', Normalized Expected Answer: '{normalized_expected_answer}'")

    # Check for an exact match
    if normalized_user_answer == normalized_expected_answer:
        return 1  # Full score for exact match

    # Check for partial match
    expected_words = set(normalized_expected_answer.split())
    user_words = set(normalized_user_answer.split())
    matching_words = expected_words.intersection(user_words)

    # Calculate partial score based on matching words
    if expected_words:  # Prevent division by zero
        matching_score = len(matching_words) / len(expected_words)
        return round(matching_score, 2)  # Return a score between 0 and 1
    else:
        return 0  # No expected words to match against



def format_bot_response(text):
    # Replace ## with <h2> for headings
    text = text.replace("##", "<h2>").replace("\n", "</h2>\n")  # Close <h2> at the end of the line
    # Replace ** with <strong> for bold text
    while '**' in text:
        start = text.index('**')
        end = text.index('**', start + 2)
        text = text[:start] + '<strong>' + text[start + 2:end] + '</strong>' + text[end + 2:]

    # Split text into lines for line-by-line processing
    lines = text.splitlines()
    formatted_lines = []

    for line in lines:
        # Handle bullet points with * and add a dot at the end of each line
        if line.strip().startswith("*"):
            formatted_line = f"<li>{line.strip()[1:].strip()}.</li>"
        else:
            # Keep other text as-is
            formatted_line = line
        formatted_lines.append(formatted_line)

    # Join lines back into a single text block with <br> for line breaks
    return "<br>".join(formatted_lines)

def strip_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()

def generate_response(user_input):
    api_key = 'AIzaSyBI5i2XujKIQhD8wIxwZ4CUAuNzzXYK-hI'  # Replace with your actual API key
    url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=' + api_key

    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": user_input}]}]}

    response = requests.post(url, headers=headers, json=data)

    # Check if the response status is OK
    if response.status_code == 200:
        response_data = response.json()

        # Ensure the structure exists before accessing the data
        if 'candidates' in response_data and len(response_data['candidates']) > 0:
            candidate = response_data['candidates'][0]

            if 'content' in candidate and 'parts' in candidate['content'] and len(candidate['content']['parts']) > 0:
                raw_text = candidate['content']['parts'][0]['text']
                formatted_response = format_bot_response(raw_text)  # Format the response text
                return strip_html(formatted_response)  # Strip HTML tags from the formatted response
            else:
                return "Error: Response format is missing 'content' or 'parts'."
        else:
            return "Error: No candidates found in response."
    else:
        return f"Error: {response.status_code} - {response.text}"

def generate_review(score):
    """Generate a review message based on the score."""
    if score >= 80:
        return "Excellent performance! You are well-prepared for the interview."
    elif score >= 60:
        return "Good job! A bit more practice would help."
    elif score >= 40:
        return "Fair performance. Consider working on your answers."
    else:
        return "Needs improvement. Focus on key areas for development."
# Dashboard route
@app.route('/')
def home():
    return render_template('dashboard.html')

# Function to analyze resume
@app.route("/resume", methods=["POST"])
def analyze_resume():
    file = request.files.get("resume")
    job_description = request.form.get("job_description")

    if not file or not job_description:
        return jsonify({"error": "Please provide both resume and job description."}), 400

    # Handle different file types
    try:
        if file.filename.endswith(".pdf"):
            resume_text = extract_text_from_pdf(file)
        elif file.filename.endswith(".docx"):
            resume_text = extract_text_from_docx(file)
        elif file.filename.endswith(".txt"):
            resume_text = file.read().decode('utf-8')
        else:
            return jsonify({"error": "Unsupported file format. Please upload a PDF, DOCX, or TXT file."}), 400
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return jsonify({"error": "Error reading the file."}), 500

    # Get predicted scores and skills found
    result = score_resume(resume_text, job_description)
    job_category = classify_job_category(resume_text)

    # Render the results page with the relevant data
    return render_template("result.html",
                           total_score=result["total_score"],
                           skill_match=result["skill_match_percentage"],
                           semantic_similarity=result["semantic_similarity"],
                           cosine_similarity=result["cosine_similarity"],
                           skills_found=result["skills_found"],
                           job_category=job_category)

@app.route('/interview', methods=['GET', 'POST'])
def interview_bot():
    # Load total questions once at the start
    total_questions = len(general_interview_questions) + sum(len(v) for v in job_specific_questions.values())

    if request.method == 'POST':
        user_answer = request.form['answer']
        question_index = int(request.form['question_index'])
        expected_answer = ""
        score = session.get('score', 0)  # Retrieve the current cumulative score from the session

        # Determine the expected answer based on the question type
        if question_index < len(general_interview_questions):  # General questions
            expected_answer = general_interview_questions[question_index]['expected_answer']
        else:  # Job-specific questions
            predicted_job = session.get('predicted_job')
            specific_question_index = question_index - len(general_interview_questions)  # Adjust index
            if predicted_job in job_specific_questions and specific_question_index < len(job_specific_questions[predicted_job]):
                expected_answer = job_specific_questions[predicted_job][specific_question_index]['expected_answer']
            else:
                # No more questions, render the final score page
                final_score = min(score, 100)  # Cap the score at 100%
                return render_template('interview_score.html', total_score=final_score,
                                       review=generate_review(final_score))

        # Calculate the current question's score
        current_question_score = compare_answers(user_answer, expected_answer)

        # Update the cumulative score correctly
        score += (current_question_score / total_questions) * 100  # Increment score based on total questions
        score = min(score, 100)  # Cap the cumulative score at 100%
        session['score'] = score  # Save updated score in session

        # Debugging output
        print(f"User Answer: {user_answer}, Expected Answer: {expected_answer}, Current Question Score: {current_question_score}, Total Score: {score}")

        # Check if there are more questions to ask
        if question_index + 1 < total_questions:
            next_question_index = question_index + 1
            if next_question_index < len(general_interview_questions):
                next_question = general_interview_questions[next_question_index]['question']
            else:
                predicted_job = session.get('predicted_job')
                specific_question_index = next_question_index - len(general_interview_questions)
                if specific_question_index < len(job_specific_questions[predicted_job]):
                    next_question = job_specific_questions[predicted_job][specific_question_index]['question']
                else:
                    # No more questions; calculate final score
                    final_score = min(score, 100)
                    return render_template('interview_score.html', total_score=final_score,
                                           review=generate_review(final_score))

            # Pass the current score (capped at 100) to the template
            return render_template('interview.html', question=next_question, question_index=next_question_index,
                                   score=score, total_questions=total_questions)  # Display the cumulative score
        else:
            # All questions answered; calculate and display final score
            final_score = min(score, 100)  # Cap score at 100%
            return render_template('interview_score.html', total_score=final_score, review=generate_review(final_score))

    # Initial question setup
    if 'predicted_job' in session:
        predicted_job = session['predicted_job']
        first_question = general_interview_questions[0]['question']
        session['score'] = 0  # Initialize score for the session
        return render_template('interview.html', question=first_question, question_index=0,
                               score=0, total_questions=total_questions)

    return "Error: Please analyze your resume first."

@app.route('/chat', methods=['GET', 'POST'] )
def chat_with_bot():
    if request.method == 'POST':
        user_input = request.form['user_input']
        bot_response = generate_response(user_input)
        return render_template('chat.html', user_input=user_input, bot_response=bot_response)
    return render_template('chat.html')

@app.route('/cfg', methods=['GET', 'POST'])
def cfg():
    if request.method == 'POST':
        input_text = request.form['input_text']
        tagged_words, parse_trees = cfg_parser(input_text)

        # Start a new thread to display the parse tree
        threading.Thread(target=run_tkinter, args=(parse_trees,)).start()

        return render_template('cfg_result.html', input_text=input_text, tagged_words=tagged_words)
    return render_template('cfg.html')

# Semantic Analysis Route
@app.route('/semantic_analysis', methods=['GET', 'POST'])
def semantic_analysis():
    if request.method == 'POST':
        job_description = request.form.get('job_description', '').strip()
        file = request.files.get('resume')

        # Check if a file was uploaded
        if file and (file.filename.endswith('.txt') or file.filename.endswith('.pdf') or file.filename.endswith('.docx')):
            try:
                # Process .txt file
                if file.filename.endswith('.txt'):
                    resume_text = file.read().decode('utf-8')

                # Process .pdf file
                elif file.filename.endswith('.pdf'):
                    resume_text = extract_text_from_pdf(file)  # Use the PDF extraction function

                # Process .docx file
                elif file.filename.endswith('.docx'):
                    resume_text = extract_text_from_docx(file)  # Use the .docx extraction function

                # Validate resume_text
                if not resume_text:
                    return "Error: The file appears to be empty or could not be read. Please upload a valid resume."

                # Check if job description was provided
                if not job_description:
                    return "Error: Job description is required for semantic analysis."

                # Calculate semantic similarity between resume and job description
                similarity_score = calculate_semantic_similarity(resume_text, job_description)

                return render_template('semantic_analysis_result.html', similarity_score=similarity_score)

            except Exception as e:
                return f"Error processing the resume: {str(e)}"
        else:
            return "Invalid file format. Please upload a .txt, .pdf, or .docx file."
    return render_template('semantic_analysis.html')


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
