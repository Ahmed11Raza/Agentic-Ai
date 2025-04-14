import streamlit as st
import fitz  # PyMuPDF
import re
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import matplotlib.pyplot as plt
import plotly.express as px

# Download necessary NLTK data at the beginning
# Fix the NLTK resource download issue
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Set of common tech skills for detection
COMMON_TECH_SKILLS = {
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'swift',
    'kotlin', 'rust', 'go', 'scala', 'r', 'dart', 'perl', 'matlab', 'sql', 'nosql',
    'mongodb', 'postgresql', 'mysql', 'oracle', 'sqlite', 'redis', 'cassandra',
    'react', 'angular', 'vue', 'svelte', 'django', 'flask', 'fastapi', 'spring',
    'node.js', 'express', 'ruby on rails', 'laravel', 'asp.net', 'tensorflow',
    'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn',
    'tableau', 'power bi', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
    'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence', 'agile', 'scrum',
    'kanban', 'ci/cd', 'elasticsearch', 'kafka', 'hadoop', 'spark', 'airflow',
    'devops', 'mlops', 'linux', 'unix', 'bash', 'powershell', 'rest', 'graphql',
    'oauth', 'jwt', 'machine learning', 'deep learning', 'nlp', 'computer vision'
}

class ResumeAnalyzer:
    def __init__(self):
        # Initialize stopwords after ensuring they're downloaded
        self.stop_words = set(stopwords.words('english'))
        
    def extract_text_from_pdf(self, pdf_file):
        """Extract text content from a PDF file."""
        try:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")
            return ""
        
    def clean_text(self, text):
        """Clean and normalize text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def extract_skills(self, text):
        """Extract skills from text."""
        # Tokenize and clean the text with error handling
        try:
            words = word_tokenize(text.lower())
        except LookupError:
            # Fallback simple tokenization if NLTK fails
            words = text.lower().split()
        
        words = [word for word in words if word.isalpha() and word not in self.stop_words]
        
        # Extract single word skills
        skills = [word for word in words if word in COMMON_TECH_SKILLS]
        
        # Extract multi-word skills
        for skill in COMMON_TECH_SKILLS:
            if ' ' in skill and skill in text.lower():
                skills.append(skill)
        
        # Count occurrences
        skill_counts = Counter(skills)
        return dict(skill_counts)
    
    def extract_education(self, text):
        """Extract education details from text."""
        education_keywords = ['bachelor', 'master', 'phd', 'doctorate', 'bsc', 'msc', 'b.tech', 'm.tech', 'degree']
        education_pattern = r'(?i)(?:' + '|'.join(education_keywords) + r')[^.]*?(?:\d{4}|\d{2})'
        education_matches = re.findall(education_pattern, text)
        return education_matches
    
    def extract_experience(self, text):
        """Extract work experience details."""
        # Look for experience sections or job titles with dates
        exp_pattern = r'(?i)(?:experience|work|employment).*?(?=\n\n|\Z)'
        experience_section = re.findall(exp_pattern, text)
        
        if not experience_section:
            # Look for job titles with years
            exp_pattern = r'(?i)(?:developer|engineer|analyst|manager|director|consultant).*?(?:\d{4}|\d{2})'
            experience_section = re.findall(exp_pattern, text)
        
        return experience_section
    
    def calculate_experience_years(self, text):
        """Estimate years of experience."""
        # Look for patterns like "X years of experience"
        year_pattern = r'(\d+)(?:\.\d+)?\s*(?:years|yrs)'
        years_matches = re.findall(year_pattern, text.lower())
        
        if years_matches:
            # Convert all matches to float and return the highest
            years = [float(y) for y in years_matches]
            return max(years)
        
        # If no direct year mention, try to infer from date ranges
        date_pattern = r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{4}\s*-\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|present|current)[a-z]*\s*\d{0,4}'
        date_ranges = re.findall(date_pattern, text.lower())
        
        total_years = 0
        current_year = 2025  # Current year
        
        for date_range in date_ranges:
            start_year_match = re.search(r'(\d{4})\s*-', date_range)
            end_year_match = re.search(r'-\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{4})', date_range)
            
            if start_year_match:
                start_year = int(start_year_match.group(1))
                
                if end_year_match:
                    end_year = int(end_year_match.group(1))
                elif 'present' in date_range or 'current' in date_range:
                    end_year = current_year
                else:
                    end_year = start_year + 1  # Assume at least 1 year
                
                total_years += (end_year - start_year)
        
        return total_years if total_years > 0 else None
    
    def calculate_similarity(self, resume_text, job_description):
        """Calculate similarity between resume and job description."""
        if not resume_text or not job_description:
            return 0
        
        # Clean texts
        cleaned_resume = self.clean_text(resume_text)
        cleaned_job = self.clean_text(job_description)
        
        # Create document vectors
        vectorizer = CountVectorizer()
        try:
            count_matrix = vectorizer.fit_transform([cleaned_resume, cleaned_job])
            similarity = cosine_similarity(count_matrix)[0][1]
            return similarity
        except:
            return 0
    
    def analyze_resume(self, resume_text, job_description=None):
        """Analyze resume and compare with job description if provided."""
        results = {}
        
        # Extract basic metrics
        word_count = len(resume_text.split())
        results['word_count'] = word_count
        
        # Extract skills
        skills = self.extract_skills(resume_text)
        results['skills'] = skills
        results['skills_count'] = len(skills)
        
        # Extract education
        education = self.extract_education(resume_text)
        results['education'] = education
        
        # Extract experience
        experience = self.extract_experience(resume_text)
        results['experience'] = experience
        
        # Estimate years of experience
        years = self.calculate_experience_years(resume_text)
        results['years_of_experience'] = years
        
        # Compare with job description if provided
        if job_description:
            # Calculate overall similarity
            similarity = self.calculate_similarity(resume_text, job_description)
            results['similarity_score'] = similarity
            
            # Extract skills from job description
            job_skills = self.extract_skills(job_description)
            results['job_skills'] = job_skills
            
            # Calculate skills match
            matching_skills = set(skills.keys()) & set(job_skills.keys())
            results['matching_skills'] = list(matching_skills)
            results['matching_skills_count'] = len(matching_skills)
            
            if job_skills:
                results['skills_match_percentage'] = len(matching_skills) / len(job_skills) * 100
            else:
                results['skills_match_percentage'] = 0
                
            # Calculate overall score (weighted average)
            if similarity > 0:
                overall_score = (
                    0.4 * similarity + 
                    0.4 * (len(matching_skills) / max(len(job_skills), 1)) + 
                    0.2 * min(1.0, (years or 0) / 5.0)  # Cap experience points at 5 years
                ) * 100
                results['overall_score'] = overall_score
            else:
                results['overall_score'] = 0
        
        return results


def display_results(results, job_description=None):
    """Display analysis results in the Streamlit app."""
    st.header("Resume Analysis Results")
    
    # Basic metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Word Count", results['word_count'])
    with col2:
        st.metric("Skills Identified", results['skills_count'])
    with col3:
        if results.get('years_of_experience'):
            st.metric("Years of Experience", f"{results['years_of_experience']:.1f}")
        else:
            st.metric("Years of Experience", "Not found")
    
    # Skills section
    st.subheader("Skills Identified")
    if results['skills']:
        # Create a DataFrame for the skills
        skills_df = pd.DataFrame({
            'Skill': list(results['skills'].keys()),
            'Count': list(results['skills'].values())
        })
        skills_df = skills_df.sort_values('Count', ascending=False)
        
        # Display top 15 skills as a bar chart
        fig = px.bar(
            skills_df.head(15), 
            x='Skill', 
            y='Count',
            title='Top Skills Mentioned in Resume'
        )
        st.plotly_chart(fig)
        
        # Display all skills in a data table
        st.dataframe(skills_df)
    else:
        st.info("No skills identified.")
    
    # Education section
    st.subheader("Education")
    if results['education']:
        for edu in results['education']:
            st.write(f"• {edu.strip()}")
    else:
        st.info("No education details extracted.")
    
    # Experience section
    st.subheader("Work Experience")
    if results['experience']:
        for exp in results['experience'][:5]:  # Limit to 5 experiences
            st.write(f"• {exp.strip()}")
    else:
        st.info("No work experience details extracted.")
    
    # Job description comparison
    if job_description:
        st.header("Job Description Comparison")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Match Score", f"{results['overall_score']:.1f}%")
        with col2:
            st.metric("Content Similarity", f"{results['similarity_score']*100:.1f}%")
        with col3:
            st.metric("Skills Match", f"{results['skills_match_percentage']:.1f}%")
        
        # Matching skills
        st.subheader("Matching Skills")
        if results['matching_skills']:
            matching_df = pd.DataFrame({
                'Skill': results['matching_skills']
            })
            st.dataframe(matching_df)
        else:
            st.info("No matching skills found.")
        
        # Missing skills
        st.subheader("Skills in Job Description Not Found in Resume")
        missing_skills = set(results['job_skills'].keys()) - set(results['skills'].keys())
        if missing_skills:
            missing_df = pd.DataFrame({
                'Skill': list(missing_skills)
            })
            st.dataframe(missing_df)
        else:
            st.success("Your resume covers all skills mentioned in the job description!")


def main():
    st.set_page_config(
        page_title="AI Resume Analyzer",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("AI Resume Analyzer")
    st.write("Upload your resume and optionally a job description to analyze your match.")
    
    analyzer = ResumeAnalyzer()
    
    # Create two columns for resume and job description uploads
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Upload Resume")
        resume_file = st.file_uploader("Choose a PDF resume", type="pdf")
    
    with col2:
        st.header("Job Description (Optional)")
        job_desc_method = st.radio(
            "How would you like to provide the job description?",
            ["Upload PDF", "Paste Text"]
        )
        
        if job_desc_method == "Upload PDF":
            job_desc_file = st.file_uploader("Choose a PDF job description", type="pdf")
            job_description = None
            if job_desc_file:
                job_description = analyzer.extract_text_from_pdf(job_desc_file)
        else:
            job_description = st.text_area("Paste job description here", height=300)
    
    if resume_file:
        with st.spinner("Analyzing resume..."):
            # Extract text from PDF
            resume_text = analyzer.extract_text_from_pdf(resume_file)
            
            if resume_text:
                # Analyze resume
                results = analyzer.analyze_resume(resume_text, job_description)
                
                # Display results
                display_results(results, job_description)
                
                # Add download option for a report
                st.download_button(
                    label="Download Analysis Report",
                    data=str(results),
                    file_name="resume_analysis_report.txt",
                    mime="text/plain"
                )
            else:
                st.error("Failed to extract text from the PDF. Please try another file.")

if __name__ == "__main__":
    main()