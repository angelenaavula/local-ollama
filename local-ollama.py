#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', 'install --q unstructured langchain')
get_ipython().run_line_magic('pip', 'install --q "unstructured[all-docs]"')


# In[2]:


from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader


# In[3]:


local_path = "The 2024 ICC Men's T20 World Cup.pdf"

# Local PDF file uploads
if local_path:
  loader = UnstructuredPDFLoader(file_path=local_path)
  data = loader.load()
else:
  print("Upload a PDF file")


# In[4]:


# Preview first page
data[0].page_content


# In[5]:


get_ipython().system('ollama pull nomic-embed-text')


# In[6]:


pip install ollama


# In[11]:


print(dir(ollama))


# In[6]:


get_ipython().system('ollama list')


# In[7]:


get_ipython().run_line_magic('pip', 'install --q chromadb')
get_ipython().run_line_magic('pip', 'install --q langchain-text-splitters')


# In[ ]:





# In[8]:


from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


# In[10]:


# Split and chunk 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)


# In[11]:


vector_db = Chroma.from_documents(
      documents=chunks,
      embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
      collection_name="local-rag",
        # Add this line
  )


# In[12]:


from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever


# In[13]:


# LLM from Ollama
local_model = "llama2"
llm = ChatOllama(model=local_model)


# In[14]:


QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to retrieve relevant
    documents from a vector database based on the user's question. Provide the most
    relevant documents that match the question.
    Question: {question}""",
)


# In[15]:


retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), 
    llm,
    prompt=QUERY_PROMPT
)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


# In[16]:


chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# In[35]:


chain.invoke(input(""))


# In[17]:


question = "who is shah rukh khan?"
result = chain.invoke({"context": data[0].page_content, "question": question})
print(result)


# In[ ]:




