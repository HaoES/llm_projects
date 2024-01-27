import requests
from newspaper import Article

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
}

article_url = "https://www.artificialintelligence-news.com/2022/01/25/meta-claims-new-ai-supercomputer-will-set-records/"

session = requests.Session()


try:
    response = session.get(article_url, headers=headers, timeout=10)

    if response.status_code == 200:
        article = Article(article_url)
        article.download()
        article.parse()

        print(f"Title: {article.title}")
        print(f"Text: {article.text}")
    else:
        print(f"Failed to fetch article at {article_url}")
except Exception as e:
    print(f"Error occurred while fetching article at {article_url}: {e}")

from langchain.schema import HumanMessage

# we get the article data from the scraping part
article_title = article.title
article_text = article.text

from langchain.output_parsers import PydanticOutputParser
from pydantic import validator
from pydantic import BaseModel, Field
from typing import List


# create output parser class
class ArticleSummary(BaseModel):
    title: str = Field(description="Title of the article")
    summary: List[str] = Field(description="Bulleted list summary of the article")

    # validating whether the generated summary has at least three lines
    @validator("summary")
    def has_three_or_more_lines(cls, list_of_lines):
        if len(list_of_lines) < 3:
            raise ValueError("Generated summary has less than three bullet points!")
        return list_of_lines


# set up output parser
parser = PydanticOutputParser(pydantic_object=ArticleSummary)

from langchain.prompts import PromptTemplate


# create prompt template
# notice that we are specifying the "partial_variables" parameter
template = """
You are a very good assistant that summarizes online articles.

Here's the article you want to summarize.

==================
Title: {article_title}

{article_text}
==================

{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["article_title", "article_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Format the prompt using the article title and text obtained from scraping
formatted_prompt = prompt.format_prompt(
    article_title=article_title, article_text=article_text
)

from langchain.llms import OpenAI


# instantiate model class
model = OpenAI(model_name="gpt-3.5-turbo", temperature=0.0)

# Use the model to generate a summary
output = model(formatted_prompt.to_string())

# Parse the output into the Pydantic model
parsed_output = parser.parse(output)
print(parsed_output)
