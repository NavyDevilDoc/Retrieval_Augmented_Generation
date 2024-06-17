import os
import time
import sys
import warnings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.chat_models import ChatOllama
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import OllamaEmbeddings, GPT4AllEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from operator import itemgetter
import torch

warnings.filterwarnings('ignore')

def format_text(input_text, n=100):
    # First pass: Add a newline after each colon
    input_text = input_text.replace(':', ':\n')
    # Second pass: Add a newline every n characters, taking the new lines into account
    formatted_text = ''
    current_length = 0  # Track the current length of the line
    for word in input_text.split(' '):  # Split the text into words
        word_length = len(word)
        if current_length + word_length > n:
            # If adding the next word exceeds the limit, start a new line
            formatted_text += '\n' + word
            current_length = word_length
        else:
            # Otherwise, add the word to the current line
            if formatted_text:  # Add a space before the word if it's not the start of the text
                formatted_text += ' '
                current_length += 1  # Account for the added space
            formatted_text += word
            current_length += word_length
        # Account for newlines within the word itself (e.g., after a colon)
        newline_count = word.count('\n')
        if newline_count > 0:
            # Reset the current length for new lines
            current_length = word_length - word.rfind('\n') - 1
    return formatted_text

def save_results_to_file(model, embedding_model, index_name, question, answer):
    filename = f"results_{embedding_model}.txt"
    with open(filename, "a") as file:
        file.write(f"Experimental Setup - Large Language Model: {model}\n")
        file.write(f"                     Embedding Model: {embedding_model}\n")
        file.write(f"                     Index: {index_name}\n")
        file.write(f"Question: {question}\n")
        file.write(f"Answer: {answer}\n")
        file.write("\n")

# Load environment variables
print("Loading environment variables...")
load_dotenv(r"C:\Users\docsp\Desktop\AI_ML_Folder\Python_Practice_Folder\Natural_Language_Processing\env_variables.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
print("Environment variables loaded successfully.")

# Define the choices for LLMs and embedding models

'''
LLM Choices
 - GPT-4o (gpt-4o)
 - GPT-3.5 Turbo (gpt-3.5-turbo)
 - Phi3 Medium (phi3:medium)
 - Llama3 8B (llama3:instruct)
 - Llama3 70B (llama3:70b-instruct) (don't use without CUDA support)
 - Mistral 7B (mistral:instruct)
 
Embedding Model Choices
- Ollama
  - nomic-embed-text
  - mxbai-embed-large
  - all-minilm
- GPT
  -text-embedding-3-small
  -text-embedding-3-large
  -text-embedding-ada-002
- Sentence Transformer (these are models pulled from Hugging Face)
  -sentence-transformers/all-MiniLM-L12-v2

Embedding Schemes
- ollama
- gpt
- sentence_transformer
'''

# User selections
selected_llm_type = "ollama"  # Can be "gpt" or "ollama"
selected_llm = "phi3:medium"
selected_embedding_scheme = "ollama"  # Can be "ollama", "gpt", or "sentence_transformer"
selected_embedding_model = "mxbai-embed-large"

# Load the model
if selected_llm_type == "gpt":
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
else:
    llm = ChatOllama(model=selected_llm, temperature=0.2, top_k=20, top_p=0.3)

'''
# Ensure model is moved to GPU if supported
if torch.cuda.is_available():
    print("Cuda is Availabe")
    llm.to('cuda')
else:
    print("Cuda Can't be found")
'''

# Load the embeddings
if selected_embedding_scheme == "gpt":
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=selected_embedding_model)
elif selected_embedding_scheme == "ollama":
    embeddings = OllamaEmbeddings(model=selected_embedding_model)
else:
    embeddings = SentenceTransformer(selected_embedding_model)

# Determine the embedding dimensions dynamically
text = "This is a text document."
if selected_embedding_scheme == "sentence_transformer":
    dimensions = len(embeddings.encode([text])[0])
else:
    dimensions = len(embeddings.embed_documents([text])[0])

# Output the choices for confirmation
print(f"Selected LLM: {selected_llm}")
print(f"Selected Embedding Scheme: {selected_embedding_scheme}")
print(f"Selected Embedding Model: {selected_embedding_model}")
print(f"Embedding dimensions: {dimensions}")


# Load and split the documents
split_documents = True
if split_documents: 
    print("Loading documents...")
    loader = PyPDFLoader(r"C:\Users\docsp\Downloads\NMED_Campaign_Plan_2028_1_5.pdf")
    text_documents = loader.load()
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = text_splitter.split_documents(text_documents)
    print(f"Done. Number of documents: {len(documents)}")

# Set up Pinecone
use_Pinecone = False
if use_Pinecone:
    print("Setting up Pinecone...")
    document_name = 'navmed-campaign-plan'
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = f'{document_name}-{selected_embedding_model}'
    print(f"Index name: {index_name}")

# Create a new index as needed
create_new_index = False
if create_new_index:
    spec = ServerlessSpec(cloud='aws', region='us-east-1')
    if index_name in pc.list_indexes().names():
        print(f"Deleting existing index: {index_name}")
        pc.delete_index(index_name)
    print(f"Creating new index: {index_name}")
    pc.create_index(index_name, 
                    dimension=dimensions,  
                    metric='cosine', 
                    spec=spec)
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
        
    print(f"Index '{index_name}' ready for use")
    index = pc.Index(index_name)
    index.describe_index_stats()

# Choose the data storage method
data_storage = 2
if data_storage == 0:
    print(f"Active Index: {index_name}")
    # Prompt the user to confirm the index name
    user_input = input("Enter 1 to continue using this index name or 2 to exit: ")
    # Check the user input
    if user_input == "2":
        print("Exiting as requested by the user.")
        sys.exit(0)
    elif user_input != "1":
        print("Invalid input. Exiting.")
        sys.exit(1)
    print(f"Uploading documents to Pinecone index {index_name}")
    datastore = PineconeVectorStore.from_documents(documents, embedding=embeddings, index_name=index_name)          
    print("Finished uploading documents to Pinecone index.")
    
elif data_storage == 1:
    print(f"Using existing Pinecone index {index_name} ")
    datastore = PineconeVectorStore.from_existing_index(index_name, embeddings)                                   
    print("Finished pulling documents from Pinecone index.")
    
elif data_storage == 2:
    print("Storing documents locally")
    datastore = DocArrayInMemorySearch.from_documents(documents, embeddings)
    print("Finished uploading documents to local storage.")

# Set up the model, parser, and prompt
parser = StrOutputParser()
template = """
Based on the context provided, answer the question with a detailed explanation. 
If the question is unclear or lacks sufficient context to provide an informed answer, respond with "I don't know" or ask for clarification. 
Spell out all acronyms. 
Provide references for major points based on the context provided. Only use information from the designated documentation.
Ensure your answer is thorough and detailed, offering insights and explanations to support your conclusions.

Context: {context}
Question: {query}
"""
prompt = PromptTemplate.from_template(template)
retriever = datastore.as_retriever()
chain = (
    {
        "context": itemgetter("query") | retriever, "query": itemgetter("query"),
    }
    | prompt
    | llm
    | parser
)

# Define the questions to ask and save the results
queries = ["Please provide a detailed report on the document. Focus specifically on the objectives, EXMED capabilities, and LOEs. Each LOE should have a dedicated paragraph summarizing key points. "]

print(f"Experimental Setup - Large Language Model: {selected_llm}\n"
      f"                     Embedding Model: {selected_embedding_model}\n")
for query in queries:
    print(f"Question: {query}\n")
    print(f"Answer: {format_text(chain.invoke({'query': query}), 100)}")
    print()
