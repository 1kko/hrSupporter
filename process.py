import ast
import logging
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dateutil import parser
from dateutil.relativedelta import relativedelta
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama.llms import OllamaLLM
from pydantic import (
    BaseModel,  # Replace the langchain_core import
    Field,
)

# setup logging for both console and file
# format is datetime, filename:linenumber, levelname, message.
file_handler = logging.FileHandler('process.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s'))

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.info("Logger initialized")

class Experience(BaseModel):
    company: str = Field(description="Name of the company")
    title: str = Field(description="Title of applicant while in the company")
    start_date: str = Field(description="Start working date in YYYY-MM format")
    end_date: str = Field(
        description="End working date in YYYY-MM format, or 'Present'"
    )
    achievements: str | List[str] | None = Field(
        description="List of key achievements or responsibilities"
    )


class ResumeExperiences(BaseModel):
    experiences: List[Experience] = Field(
        description="List of work and educational experiences"
    )


def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime object."""
    if date_str.lower() in ["present", "현재"]:
        return datetime.now()
    if date_str.lower() in ["unknown", "미상"]:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m")
    except ValueError:
        return parser.parse(date_str)


def calculate_gap(end_date: datetime, start_date: datetime) -> Optional[str]:
    """Calculate the gap between two dates."""
    if end_date < start_date:
        diff = relativedelta(start_date, end_date)
        months = diff.years * 12 + diff.months
        if months <= 1:
            return None
        elif months < 12:
            return f"{months} months gap"
        else:
            years = months // 12
            remaining_months = months % 12
            if remaining_months == 0:
                return f"{years} years gap"
            return f"{years} years, {remaining_months} months gap"
    return None


class ResumeProcessor:
    def __init__(self, model="llama3.1"):
        self.llm = OllamaLLM(model=model)
        self.parser = PydanticOutputParser(pydantic_object=ResumeExperiences)

        # Create prompt template
        self.prompt_timeline = PromptTemplate(
            template="""You are an expert and experienced HR firm.
            Language of expected output should be same to the resume.
            You are analyzing resume and extracting work experiences and school years.
            You need to extract these information from the resume text and format them as JSON.
            These information is scattered in resume, so you need to extract them all. Some of them are duplicates, so you need to remove them.
            Required fields: company(or school name), title, start_date, end_date, achievements.
            
            Rules:
            1. Dates must be in YYYY-MM format, where YYYY is year, and MM for month
               Current year and month is 2024-12. Month must be between 01 and 12.
            2. Use "Present" for current positions
            3. If company name is missing, use "Unknown"
            4. for same start date and same company, keep them in one row.
            5. If title is missing, use "Unknown"
            6. don't translate keywords and company names, keep it in original language.
            7. Remove any newlines if it effects to output json.
            8. because we're using markdown, use <br> for new line of achievements.
            
            {format_instructions}
            
            Resume text:
            {text}
            """,
            input_variables=["text"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )
        
        self.overall_resume_commenter = PromptTemplate(
            template="""You are interviewer of our company, to evaluate applicant's resume.
            Language of expected output should be same to the resume.
            If you find typos or grammatical errors, list them.
            If you find any inconsistencies in the resume, list them.
            make your evaluation report concise and clear. bullet points are good.
            No need to comment how to improve the resume, just list the issues.
            Output should be markdown format.
            Base on the resume, you need to evaluate and score (1-5) the interviewee's skills and experiences considering his/her background and experience in the following categories:
            
            - Technical Skills
            - Experience
            - Education
            - Projects
            - Achievements
            
            Resume text:
            {text}
            """,
            input_variables=["text"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )


        self.questionnaire_generator = PromptTemplate(
            template="""You are interviewer of our company, to evaluate applicant's resume.
            Language of expected output should be same to the resume.
            As a interviewer, create questions that are relevant to the resume.
            Make max 5 questions for each category, his/her technical skills, experience, education, projects, achievements.
            For technical questions, be direct and clear. 
            For other types of questions, be indirect and subtle to find out the interviewee is not lying.
            All questions should not be duplicate. if number of questions is less than 5, make it more.
            Output should be markdown format.
            """,
            input_variables=["text"],
        )

    def load_and_split_pdf(self, pdf_path: str) -> List[str]:
        """Load PDF and split into chunks."""
        logger.info(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logger.debug(f"Loaded PDF: {pdf_path}")
        
        # each document is a page of pdf, so we don't need to split them
        return documents

    def process_timeline_chunk(self, chunk) -> List[Experience]:
        """Process a single chunk of text."""
        prompt = self.prompt_timeline.format(text=chunk.page_content)

        try:
            output = self.llm.invoke(prompt)

            try:
                # Clean the output string
                json_match = re.search(r"\{.*\}", output, re.DOTALL)
                if json_match:
                    json_str = json_match.group()

                    # Parse JSON
                    # Originally it was loading JSON, but it was not working well (output sometimes has trailing comma), so I changed to ast.literal_eval
                    data = ast.literal_eval(json_str)
                    experiences = data.get("experiences", data.get("experience", []))

                    normalized_experiences = []
                    for exp in experiences:
                        # Create normalized experience
                        normalized_exp = {
                            "company": exp.get("company", "Unknown"),
                            "title": exp.get("title", "Unknown"),
                            "start_date": exp.get("start_date", ""),
                            "end_date": exp.get("end_date", ""),
                            "achievements": exp.get(
                                "achievements", ["No achievements listed"]
                            ),
                        }

                        try:
                            experience = Experience(**normalized_exp)
                            normalized_experiences.append(experience)
                        except Exception as e:
                            logger.error(f"Error creating Experience object: {e}")
                            logger.error(f"Data: {normalized_exp}")
                            continue

                    return normalized_experiences

            except Exception as e:
                logger.exception(f"Error processing JSON: {e}")
                logger.exception(f"Raw output: {output}")
                return []

        except Exception as e:
            logger.exception(f"Error invoking LLM: {e}")
            return []

    def create_timeline(self, experiences: List[Experience]) -> str:
        """Create a timeline of experiences with gaps."""
        # Sort experiences by start date
        sorted_experiences = sorted(
            experiences, key=lambda x: parse_date(x.start_date), reverse=True
        )

        timeline = "# Professional and Educational Timeline\n\n"
        timeline += (
            "| start date | end date | Organization | Role | Achievements |\n"
        )
        timeline += (
            "|------------|----------|--------------|------|--------------|\n"
        )

        for i, exp in enumerate(sorted_experiences):
            # Format achievements
            achievements = exp.achievements
            if isinstance(exp.achievements, list):
                achievements = "<br>".join([f"• {ach}" for ach in exp.achievements])
            
            achievements = achievements.replace('\n', '<br>').replace("\r", "") if achievements else '-'

            # Add experience row
            timeline += f"| {exp.start_date} | {exp.end_date} | {exp.company} | {exp.title} | {achievements} |\n"

            # Check for gaps
            if i < len(sorted_experiences) - 1:
                current_end = parse_date(exp.end_date)
                next_start = parse_date(sorted_experiences[i + 1].start_date)

                gap = calculate_gap(current_end, next_start)
                if gap:
                    timeline += f"| **{gap}** | | | | |\n"

        return timeline
    
    def process_resume_commenter_chunk(self, chunk) -> str:
        """Create a summary of experiences."""
        prompt = self.overall_resume_commenter.format(text=chunk.page_content)
        return self.llm.invoke(prompt)
    
    def process_questionnaire_chunk(self, chunk) -> str:
        """Create questionnaire."""
        prompt = self.questionnaire_generator.format(text=chunk.page_content)
        return self.llm.invoke(prompt)

    def process_resume(self, source_dir: str="source", output_dir: str="output", processed_dir:str="processed") -> None:
        """Process all PDF files in source directory and save timelines to output directory."""
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(processed_dir).mkdir(parents=True, exist_ok=True)
        
        # Get all PDF files in source directory
        pdf_files = [f for f in os.listdir(source_dir) if f.lower().endswith(".pdf")]
        logger.info(f"Found {len(pdf_files)} PDF files in {source_dir}")
        for pdf_file in pdf_files:
            start_time = time.time()
            pdf_path = os.path.join(source_dir, pdf_file)
            output_file = os.path.join(output_dir, f"{Path(pdf_file).stem}.md")

            try:
                chunks = self.load_and_split_pdf(pdf_path)
                all_experiences = []
                all_comments = ["# Resume Comment\n\n"]
                all_questionnaires = []
                for i, chunk in enumerate(chunks):
                    logger.info(f"Processing Timeline Formatter chunk {i+1}/{len(chunks)}")
                    experiences = self.process_timeline_chunk(chunk)
                    all_experiences.extend(experiences)
                
                for i, chunk in enumerate(chunks):
                    logger.info(f"Processing Resume Commenter chunk {i+1}/{len(chunks)}")
                    comment = self.process_resume_commenter_chunk(chunk)
                    all_comments.append(comment)
                    
                for i, chunk in enumerate(chunks):
                    logger.info(f"Processing Questionnaire chunk {i+1}/{len(chunks)}")
                    comment = self.process_questionnaire_chunk(chunk)
                    all_questionnaires.append(comment)

                # Remove duplicates based on company and start_date
                unique_experiences = {
                    (exp.company, exp.start_date): exp for exp in all_experiences
                }.values()


                # Save timeline to output directory
                timeline = self.create_timeline(list(unique_experiences))
                # save comment to output directory
                comments = "\n".join(all_comments)
                questionnaires = "\n".join(all_questionnaires)
                with open(output_file, "w") as f:
                    logger.debug(f"{timeline=}")
                    f.write(timeline)
                    logger.info(f"Saved timeline to {output_file}")
                    f.write("\n\n")
                    logger.debug(f"{comments=}")
                    f.write(comments)
                    logger.info(f"Saved comment to {output_file}")
                    f.write("\n\n")
                    logger.debug(f"{questionnaires=}")
                    f.write(questionnaires)
                    logger.info(f"Saved questionnaire to {output_file}")

                logger.info(f"Processed {pdf_file} successfully!")
                end_time = time.time()
                
                # move the pdf file to processed directory
                shutil.move(pdf_path, os.path.join(processed_dir, pdf_file))
                logger.info(f"Time taken for {pdf_file}: {end_time - start_time} seconds")

            except Exception as e:
                logger.exception(f"Error processing {pdf_file}: {e}")



def main() -> None:
    model = "gemma2:9b"
    logger.info(f"Loading LLM: {model}")
    processor = ResumeProcessor(model=model)

    try:
        # get total time
        start_time = time.time()
        logger.info("Processing resumes")
        processor.process_resume("source", "output")
        logger.info("All resumes have been processed!")
        end_time = time.time()
        logger.info(f"Total time: {end_time - start_time} seconds")

    except Exception as e:
        logger.exception(f"Error processing resumes: {e}")


if __name__ == "__main__":
    main()
