# IMPORTS

import os
import syslog
import time
import ssl
import smtplib
import uuid
from email.mime.text import MIMEText
import pandas as pd
import lancedb
from langchain_community.vectorstores import LanceDB
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from imap_tools import MailBox
from dotenv import load_dotenv

load_dotenv() 

# CONSTANTS
K_FACTOR = 3
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
WEIGHT = 0.2
PORT = 25
INTERVAL = 20

syslog.syslog("Program start.")

def route_to_chain(route_name):
    if "no questions" != route_name.lower():
        return answer_chain
    return "No"

def route_to_chain_two(route_name):
    if "not answerable with context" != route_name.lower():
        return question_chain
    return "No"

def create_DB():
    table = db.create_table(
        "pandas_docs",
        data=[
            {
                "vector": embedding.embed_query("MSCS"),
                "text": "MSCS",
                "id": "1",
            },
            {
                "vector": embedding.embed_query("GRE"),
                "text": "GRE",
                "id": "2",
            },
            {
                "vector": embedding.embed_query("Application"),
                "text": "Application",
                "id": "3",
            },
            {
                "vector": embedding.embed_query("Admission"),
                "text": "Admission",
                "id": "4",
            },
            {
                "vector": embedding.embed_query("Computer Science"),
                "text": "Computer Science",
                "id": "5",
            },
            {
                "vector": embedding.embed_query("Course"),
                "text": "Course",
                "id": "6",
            },
            {
                "vector": embedding.embed_query("Graduation"),
                "text": "Graduation",
                "id": "7",
            },
            {
                "vector": embedding.embed_query("Rutgers"),
                "text": "Rutgers",
                "id": "8",
            }
        ],
        mode="overwrite",
    )

def create_folders(mailbox):
    if not mailbox.folder.exists('INBOX.answered'):
        mailbox.folder.create('INBOX.answered')
    if not mailbox.folder.exists('INBOX.irrelevent'):
        mailbox.folder.create('INBOX.irrelevent')
    if not mailbox.folder.exists('INBOX.unanswerable'):
        mailbox.folder.create('INBOX.unanswerable')

# API KEYS
    
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_AI_KEY")

# LOAD DATA

df = pd.read_csv("./content/context.csv")
loader = DataFrameLoader(df, page_content_column="data")
docs = loader.load()

embedding = OpenAIEmbeddings()

# SPLIT SENTENCES

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# CREATE BM25 SPARSE KEYWORD MATCHING RETRIEVER

bm25_retriever = BM25Retriever.from_documents(all_splits)
bm25_retriever.k = K_FACTOR

# SETTING UP HYBRID SEARCH

ALL_TEXT = " ".join([doc.page_content for doc in all_splits])

# CREATE LANCEDB VECTOR STORE FOR DENSE SEMANTIC SEARCH/RETRIEVAL

syslog.syslog("LanceDB.")
db = lancedb.connect("/tmp/lancedb"+str(os.getuid()))
# create_DB()
docsearch = LanceDB.from_texts(ALL_TEXT, embedding, connection=db)
retriever_lancedb = docsearch.as_retriever(search_kwargs={"k": K_FACTOR})

# GENERATE EMBEDDINGS

retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, retriever_lancedb], weights=[WEIGHT, (1-WEIGHT)]
)

# GEMINI

llm = ChatGoogleGenerativeAI(model="gemini-pro")

# PROMPT TEMPLATES

TEMPLATE = """
  Given the input below, return a list of questions that are being asked. If there are no questions being asked, classify it as 'no questions'.

  INPUT:
  {query}

  Classification:
  """

TEMPLATE_TWO = """
  Given the input below, return a list of questions that are answerable with the context provided. If there are no questions
  that can be answered with the given context, classify it as 'not answerable with context'.

  INPUT:
  {query}

  CONTEXT:
  {context}

  Classification:
  """

TEMPLATE_THREE = """
  Given the input below, answer the questions that are being asked with the provided context. If there are no questions that can be 
  answered with the given context, classify it as 'no answer'.

  INPUT:
  {query}

  CONTEXT:
  {context}

  ANSWER:
"""

# PROMPTS

prompt_one = PromptTemplate.from_template(TEMPLATE)
prompt_two = PromptTemplate(input_variables=["query", "context"], template=TEMPLATE_TWO)
prompt_three = PromptTemplate(input_variables=["query", "context"], template=TEMPLATE_THREE)

def format_docs(documents):
    """Formats documents."""
    return "\n\n".join(doc.page_content for doc in documents)

# CHAINS

select_chain = (
    prompt_one
    | llm
    | StrOutputParser()
)
answer_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt_two
    | llm
    | StrOutputParser() 
)
rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt_three
    | llm
    | StrOutputParser()
)

answer_chain = RunnableParallel(
    {"context": retriever, "query": RunnablePassthrough()}
).assign(answer=answer_chain_from_docs)

question_chain = RunnableParallel(
    {"context": retriever, "query": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

print("Answering.")

with open("./content/question.txt", "r") as pFile:
    pLines = [
        # strip() - Removes leading/trailing whitespace.
        line.strip()
            # readlines() - Reads all the lines of a file an returns them as a list.
            for line in pFile.readlines()]
cnt = 1
for line in pLines:
  if line.strip() != "":
    print(cnt)
    print(line)
    print("")
    route_one = select_chain.invoke(line)
    print("\n" + route_one + "\n")
    chain_one = route_to_chain(route_one)
    if chain_one == "No":
        print("Not a question.")
    else:
        arr = route_one.split("\n")
        for question in arr:
            if question.strip() != "Questions:" and question.strip() != "":
                print(question)
                final = question_chain.invoke(question)
                print(final['answer'])
                print("")
    print("")
    cnt += 1
    
