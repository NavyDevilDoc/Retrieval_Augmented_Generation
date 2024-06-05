import os
from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

app = Flask(__name__)

folder_path = "datastore"

cached_llm = Ollama(model="llama3")

embeddings = OllamaEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant good at searching docuemnts. 
    If you do not have an answer from the provided information say "I don't have that information."
    Cite your sources.  [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)

@app.route("/")
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Model Flask Interface</title>
        <script>
            function sendJSON(url) {
                event.preventDefault();  // Prevent the default form behavior
                var input = document.getElementById('query_' + url).value;  // Get the input from the form
                fetch('/' + url, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ query: input })  // Convert the form data to JSON
                }).then(response => response.json())
                .then(data => {
                    document.getElementById('response_text').textContent = JSON.stringify(data, null, 2);  // Display the response
                })
                .catch(error => console.error('Error:', error));
            }

            function loadPDFs() {
                fetch('/list_pdfs')  // Assuming you have a Flask endpoint to list PDFs
                    .then(response => response.json())
                    .then(data => {
                        var list = document.getElementById('pdf_list');
                        list.innerHTML = '';  // Clear existing list
                        data.forEach(function(pdf) {
                            var item = document.createElement('li');
                            item.textContent = pdf;
                            list.appendChild(item);
                        });
                    })
                    .catch(error => console.error('Error:', error));
            }

            document.addEventListener('DOMContentLoaded', function() {
                loadPDFs();  // Load PDFs when the page loads
            });
        </script>
    </head>
    <body>
        <h1>Prototype Web-based Retrieval-augmented Generation Model</h1>

        <h2>Ask AI</h2>
        <form onsubmit="sendJSON('ai');">
            <label for="query_ai">Enter your query:</label><br>
            <input type="text" id="query_ai" name="query"><br><br>
            <input type="submit" value="Submit">
        </form>

        <h2>Ask from PDF</h2>
        <form onsubmit="sendJSON('ask_pdf');">
            <label for="query_ask_pdf">Enter your query for PDF documents:</label><br>
            <input type="text" id="query_ask_pdf" name="query"><br><br>
            <input type="submit" value="Submit">
        </form>

        <h2>Response:</h2>
        <pre id="response_text"></pre> <!-- Text area for displaying the response -->

        <h2>Upload PDF</h2>
        <form action="/pdf" method="post" enctype="multipart/form-data">
            <label for="file">Upload a PDF file:</label><br>
            <input type="file" id="file" name="file"><br><br>
            <input type="submit" value="Upload PDF" onclick="setTimeout(loadPDFs, 2000);">
        </form>

        <h2>Loaded PDF Documents:</h2>
        <ul id="pdf_list"></ul> <!-- List to display loaded PDFs -->

    </body>
    </html>
    """

@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    response = cached_llm.invoke(query)

    print(response)

    response_answer = {"answer": response}
    return response_answer


@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("Post /ask_pdf called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    print("Loading vector store")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embeddings)

    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    result = chain.invoke({"input": query})

    print(result)

    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_content": doc.page_content}
        )

    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer

@app.route("/list_pdfs", methods=["GET"])
def list_pdfs():
    pdf_directory = 'pdf'  # Directory where PDFs are stored
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    return jsonify(pdf_files)


@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    save_dir = "pdf"
    save_path = os.path.join(save_dir, file_name)
    
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save the file
    file.save(save_path)
    print(f"filename: {file_name}")

    loader = PDFPlumberLoader(save_path)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=folder_path
    )

    vector_store.persist()

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return response

def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ == "__main__":
    start_app()