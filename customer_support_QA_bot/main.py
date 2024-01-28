from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.document_loaders import SeleniumURLLoader
from langchain import PromptTemplate

# we'll use information from the following articles
urls = [
    "https://beebom.com/what-is-nft-explained/",
    "https://beebom.com/how-delete-spotify-account/",
    "https://beebom.com/how-download-gif-twitter/",
    "https://beebom.com/how-use-chatgpt-linux-terminal/",
    "https://beebom.com/how-delete-spotify-account/",
    "https://beebom.com/how-save-instagram-story-with-music/",
    "https://beebom.com/how-install-pip-windows/",
    "https://beebom.com/how-check-disk-usage-linux/",
]

# use selenium to load documents
loader = SeleniumURLLoader(urls=urls)
docs_not_splitted = loader.load()

# we split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(docs_not_splitted)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

my_activeloop_org_id = "haoes"
my_activeloop_dataset_name = "customer_supp"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# add documents to db
db.add_documents(docs)


# let's write a prompt for a customer support chatbot that
# answer questions using information extracted from our db
template = """You are an exceptional customer support chatbot that gently answers questions.

You know the following context information.

{chunks_formatted}

Answer to the following question from a customer. Use only information from the previous context information. Do not invent stuff.

Question: {query}

Answer:"""

prompt = PromptTemplate(
    input_variables=["chunks_formatted", "query"], template=template
)

# the full pipeline
#
# user question
query = "How to check disk usage in Linux?"

# retrieve relevant chunks
docs = db.similarity_search(query)
retrieved_chunks = [doc.page_content for doc in docs]

# format the prompt

chunks_formatted = "\n\n".join(retrieved_chunks)
prompt_formatted = prompt.format(chunks_formatted=chunks_formatted, query=query)

# generate answer
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)
answer = llm(prompt_formatted)
print(answer)
