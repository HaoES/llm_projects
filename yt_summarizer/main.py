import yt_dlp


def dl_mp4_from_yt(url):
    # Options for the download
    filename = "vid.mp4"
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "outtmpl": filename,
        "quiet": True,
    }

    # Download the video file.
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)


url = "https://www.youtube.com/watch?v=mBjPyte2ZZo"
dl_mp4_from_yt(url)

import whisper

model = whisper.load_model("base")
result = model.transcribe("vid.mp4")

with open("text.txt", "w") as file:
    file.write(result["text"])

from langchain import OpenAI, LLMChain
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
)

from langchain.docstore.document import Document

with open("text.txt") as f:
    text = f.read()

texts = text_splitter.split_text(text)
docs = [Document(page_content=t) for t in texts[:4]]

from langchain.chains.summarize import load_summarize_chain
import textwrap


prompt_template = """Write a concise bullet point summary of the following:


{text}


CONSCISE SUMMARY IN BULLET POINTS:"""

BULLET_POINT_PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

chain = load_summarize_chain(llm, chain_type="refine", prompt=BULLET_POINT_PROMPT)

output_summary = chain.run(docs)
wrapped_text = textwrap.fill(
    output_summary, width=1000, break_long_words=False, replace_whitespace=False
)
print(wrapped_text)
