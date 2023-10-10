from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import sys
import langchain
# import faiss

DB_FAISS_PATH = r"C:\Users\sohan\Desktop\AI_Projects\Chat_with_CSVdata-Llama-2\vectorstore\db_faiss"
loader = CSVLoader(file_path=r"C:\Users\sohan\Desktop\AI_Projects\Chat_with_CSVdata-Llama-2\data\2019.csv")
data = loader.load()
# print(data)

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)
# print(len(text_chunks))

# Download Sentence Transformers Embeddings from Hugging Face
embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Converting the text chunks into embeddings and saving the embeddings into FAISS Knowledge Base
docsearch = FAISS.from_documents(text_chunks, embeddings)
docsearch.save_local(DB_FAISS_PATH)

# query = "What is the value of GDP per capita of Finland provided in the data?"
# docs = docsearch.similarity_search_with_score(query,k=3)
# print("Result",docs)

llm = CTransformers(model=r"C:\Users\sohan\Desktop\AI_Projects\Chat_with_CSVdata-Llama-2\models\llama-2-7b-chat.Q4_K_M.gguf",
                    model_type="llama",
                    max_new_tokens=512,
                    temperature=0.1)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())

while True:
    chat_history=[]
    query = input(f"Input Prompt: ")
    if query == 'exit':
        print('Exiting')
        sys.exit()
    if query == '':
        continue

    result = qa({"question": query, "chat_history": chat_history})
    print("Response: ", result['answer'])

