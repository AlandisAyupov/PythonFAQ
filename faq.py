# IMPORTS

import os
import imaplib
import time
import os
import ssl
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
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
INTERVAL = 5

print("Program start.")

def route_to_chain(route_name):
    if "no questions" != route_name.lower():
        return answer_chain
    return "No"

def route_to_chain_two(route_name):
    if "not answerable with context" != route_name.lower():
        return question_chain
    return "No"

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

bm25_retriever = BM25Retriever.from_documents(all_splits)
bm25_retriever.k = K_FACTOR

# SETTING UP HYBRID SEARCH

ALL_TEXT = " ".join([doc.page_content for doc in all_splits])
db = lancedb.connect("/tmp/lancedb"+str(os.getuid()))
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
        }
    ],
    mode="overwrite",
)
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
  Given the input below, return a list of questions that are answerable with the context provided. If there are not questions
  that can be answered with the given context, classify it as 'not answerable with context'.

  INPUT:
  {query}

  CONTEXT:
  {context}

  Classification:
  """

TEMPLATE_THREE = """
  Respond with the questions that are being asked in the input, and answer them with the given context

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

# CONNECT

SMTP_SERVER = 'mx.farside.rutgers.edu'
server = smtplib.SMTP(SMTP_SERVER, PORT)
server.starttls()

# MAIL - Credit to stackoverflow user https://stackoverflow.com/questions/5632713/getting-n-most-recent-emails-using-imap-and-python

context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.set_ciphers('DEFAULT@SECLEVEL=1')
while True:
    with MailBox('rci.rutgers.edu', ssl_context=context).login(os.getenv("EMAIL"), os.getenv("PASS"), 'INBOX') as mailbox:
        if not mailbox.folder.exists('INBOX.answered'):
            mailbox.folder.create('INBOX.answered')
        if not mailbox.folder.exists('INBOX.irrelevent'):
            mailbox.folder.create('INBOX.irrelevent')
        if not mailbox.folder.exists('INBOX.unanswerable'):
            mailbox.folder.create('INBOX.unanswerable')
        for msg in mailbox.fetch():
            SENDER_EMAIL = os.getenv('SENDER_EMAIL')
            RECEIVER_EMAIL = msg.from_
            message = MIMEMultipart()
            message['From'] = SENDER_EMAIL
            message['To'] = RECEIVER_EMAIL
            message['Subject'] = msg.text
            print(msg.flags)
            print(msg.text)
            route_one = select_chain.invoke(msg.text)
            chain_one = route_to_chain(route_one)
            if chain_one == "No":
                print("Not a question.")
                mailbox.move(msg.uid, 'INBOX.irrelevent')
            else:
                print("Is a question.")
                route_two = answer_chain.invoke(msg.text)
                chain_two = route_to_chain_two(route_two['answer'])
                if chain_two == "No":
                    print("Not answerable with context.")
                    mailbox.move(msg.uid, 'INBOX.unanswerable')
                else:
                    print("Answerable with context.")
                    BODY = str(question_chain.invoke(msg.text)['answer'])
                    message.attach(MIMEText(BODY, 'plain'))
                    # Send the email
                    server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, message.as_string())
                    mailbox.move(msg.uid, 'INBOX.answered')
    time.sleep(INTERVAL)
    
