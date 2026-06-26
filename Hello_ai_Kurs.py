from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (PromptTemplate,
                                    HumanMessagePromptTemplate,
                                    ChatPromptTemplate)
from langchain_core.runnables import RunnablePassthrough

from langchain_openai import (ChatOpenAI,
                              OpenAIEmbeddings)

from langchain_chroma.vectorstores import Chroma

loader_pdf = PyPDFLoader("a.pdf")
docs_list = loader_pdf.load()
token_splitter = TokenTextSplitter(encoding_name="cl100k_base",
                                   chunk_size=200,
                                   chunk_overlap=40)
docs_list_tokens_split = token_splitter.split_documents(docs_list)

embedding = OpenAIEmbeddings(model='text-embedding-3-small',
                             api_key="{Open-AI-Key hier einfügen}")
vectorstore = Chroma.from_documents(documents = docs_list_tokens_split,
                                    embedding = embedding,
                                    persist_directory = "./intro-to-ai")
retriever = vectorstore.as_retriever(search_type = 'mmr',
                                     search_kwargs = { 'k':1,
                                                       'lambda_mult':0.7})
print(retriever.invoke("Eine Frage zu a.pdf stellen!")[0].page_content)



