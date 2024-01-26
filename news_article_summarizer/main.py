from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import requests
from newspaper import Article

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
}
article_url = "https://www.gamingbible.com/news/platform/steam/elden-ring-expansion-quietly-teased-mysterious-new-update-300293-20240116"

session = requests.Session()

try:
    response = session.get(article_url, headers=headers, timeout=10)

    if response.status_code == 200:
        article = Article(article_url)
        article.download()
        article.parse()

    else:
        print(f"Failed to fetch article at {article_url}")
except Exception as e:
    print(f"Error occurred while fetching article at {article_url}: {e}")


# article data
article_title = article.title
article_text = article.text

# prepare template for prompt
template = """You are an advanced AI assistant that summarizes online articles into bulleted lists.

Here's the article you want to summarize.


==================
Title: {article_title}


{article_text}
==================


Now, provide a summarized version of the article in a bulleted list format.
"""

# format prompt
prompt = template.format(article_title=article.title, article_text=article.text)


# load the model
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# generate summary
summary = chat([HumanMessage(content=prompt)])
print(summary.content)
