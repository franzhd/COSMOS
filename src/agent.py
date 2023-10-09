import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.schema.messages import SystemMessage
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.tools import Tool
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA

import openai
from dotenv import load_dotenv

from src.semantic_search import SemanticSearch

load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")




class Agent:
    def __init__(self,dummy=False, openai_api_key: str = None, model='gpt-4') -> None:
        # if openai_api_key is None, then it will look the enviroment variable OPENAI_API_KEY
        

        # Non vengono usati attualmente.
        if dummy:
            self.dummy = dummy
        else:
            self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)

            self.llm = ChatOpenAI(temperature=0, model=model, openai_api_key=openai_api_key)


            self.document_uploaded = False
            self.agent = None
            
            self.chat_history = None
            self.chain = None
            self.db = None
            self.docs_filepaths = []
            self.docs_file_basenames = []
            self.documents_dict = {}
            self.split_docs = []
            self.document_retriever_dict = {}

            self.tools = []
        
    def ask(self, question: str) -> str:
        if self.document_uploaded == False:
            response = "Please, add a document."
        else:
            response = self.agent.run(input=question)

        return response

    
    def ingest(self, file_path: os.PathLike) -> None:
        
        print(f"Path: {file_path}")
        self.docs_filepaths.append(file_path)
        file_basename = os.path.basename(file_path)
        self.docs_file_basenames.append(file_basename)
        
        # Load the document
        loader = PyPDFLoader(file_path)
        document = loader.load()
        print('loaded document')
        #Add the document to the documents_dict
        file_key = str(os.path.basename(file_basename)).split(".")[0]
        self.documents_dict[file_key] = document
        print('Add the document to the documents_dict')
        # Split the document
        splitted_documents = self.text_splitter.split_documents(document)
        self.file_names = [os.path.basename(doc) for doc in self.docs_filepaths]
        print('Split the document')
        
        # Add the splitted documents to the split_docs list
        self.split_docs.append(splitted_documents)
        print('Add the splitted documents to the split_docs list')
        self.document_uploaded = True
        print('self.document_uploaded = True')
    

        for file in self.docs_file_basenames:
            file_key = str(os.path.basename(file)).split(".")[0]
            
            recommender = SemanticSearch()
            splitted_docs = [doc.page_content
                             .replace('\xa0', ' ')
                             .replace('\xad', '')
                             .replace('\n', ' ') for doc in splitted_documents]
            
            recommender.fit(splitted_docs, batch=1024, n_neighbors=4)
            self.document_retriever_dict[file_key] = recommender
            
            self.tools += [
                Tool(
                    name="semantic_search_" + file_key,
                    description=f"Useful when you want to answer questions about {file_key}",
                    func= lambda text: self.semantic_search(text, file_key)
                ),
                Tool(  
                    name="list_text_parts_about_word_" + file_key,  
                    description="Returns a string with the text parts containing the input word. Use this if the semantic search it's not retrieving anything useful.",
                    func=lambda word: self.list_text_parts_containing_word(word, file_key),
                        ),
                
                # Tool(
                #     name="summarize_documents_" + file_key,
                #     description="Summarize the uploaded documents. The input parameter is the document you want to summarize.",
                #     func=lambda x: self.summarize_documents(file_key),
                # ),

            ]
            
        self.agent_init()
        
    def agent_init(self) -> None:

        base_prompt = """
        The user is trying to understand if the document contains conflicting information. 
        Use the given tools to find information about the question and then compare all the pieces and tell the user if there is any conflicting information.
        In case there is, tell the user the page number and the exact piece of document that contains the conflicting information and why it is conflicting.
        """
    
        for file_name in self.docs_filepaths:
            base_prompt += f"- {os.path.basename(file_name)}\n"
            
        self.system_message = SystemMessage(
            content=base_prompt
        )
        
        self.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)  
        self.agent = initialize_agent(
            tools=self.tools, 
            llm=self.llm, 
            agent=AgentType.OPENAI_MULTI_FUNCTIONS, 
            verbose=True, 
            memory=self.memory,
            agent_kwargs={  
                "system_message": self.system_message,
                "extra_prompt_messages": [MessagesPlaceholder(variable_name='chat_history')]
            }  
        )
        # self.agent = create_conversational_retrieval_agent(self.llm, self.tools, system_message=self.system_message, verbose=True)
        
    def semantic_search(self, text: str = "", document: str = "") -> str:
        recommender = self.document_retriever_dict[document]
        topn_chunks = recommender(text)
        result = ""
        result += 'Search results:\n\n'
        for c in topn_chunks:
            result += c + '\n\n'
        
        return result
    
    
    # Tool
    def list_text_parts_containing_word(self, word: str = "", document: str = "") -> str:
        """
        Returns a string with the text parts containing the input word.
        """
        parts = []
        
        doc = self.documents_dict[document]
        splitted_documents = [text.page_content for text in self.text_splitter.split_documents(doc)]
        
        for text in splitted_documents:
            if word in text:
                parts.append(text)
                
        return "\n\n".join(parts)[:4000]
        
    # Tool
    def summarize_specific_word(self, word: str= "", document: str = "") -> str:
        
        document = self.documents_dict[document]
        texts_about_word = self.semantic_search(word, document)
        llm = OpenAI(temperature=0)
        prompt= """
        Those are texts found in the document about the word {word}. Summarize it in a proper way:        
        {texts_about_word}
        """
        summarized_word = llm(prompt)
        return summarized_word
           
    # Tool 
    # def summarize_documents(self, document = ""):
    #     document = self.documents_dict[document]
    #     splitted_documents = self.text_splitter.split_documents(document)
    #     llm = OpenAI(temperature=0)
    #     chain = load_summarize_chain(llm=llm, 
    #                                 chain_type="map_reduce",
    #                                 verbose = False)
    #     print("Starting summarization")        
    #     output_summary = chain.run(splitted_documents)
    #     return output_summary
    def summarize_documents(self, document=""):
        document = self.documents_dict[document]
        splitted_documents = self.text_splitter.split_documents(document)
        llm = OpenAI(temperature=0)
        chain = load_summarize_chain(llm=llm, 
                                    chain_type="map_reduce",
                                    verbose=False)
        
        output_summaries = []
        print("Starting summarization")        
        
        for doc in splitted_documents:
            # Further split each document into smaller chunks (e.g., by paragraphs or a certain length)
            chunks = self.further_split(doc)
            
            chunk_summaries = []
            for chunk in chunks:
                # Summarize each chunk
                summary = chain.run([chunk])  # Since chain.run expects a list, we wrap chunk in a list
                chunk_summaries.extend(summary)
            
            # Combine chunk summaries to form a single summary for the document
            final_summary = " ".join(chunk_summaries)
            output_summaries.append(final_summary)

        return output_summaries
        
    def forget(self) -> None:
        self.db = None
        self.chain = None
        self.chat_history = None

    def further_split(document, max_length=4000):
        """Breaks the document into smaller chunks based on max_length."""
        return [document[i:i+max_length] for i in range(0, len(document), max_length)]