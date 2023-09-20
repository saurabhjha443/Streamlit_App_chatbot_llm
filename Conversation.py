from langchain.document_loaders import WebBaseLoader
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
# %%
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.chains import ConversationChain
# %%
from langchain.llms import GPT4All
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
import streamlit as st
from streamlit_chat import message
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import re
import urllib.parse
# >>> base = 'https://www.example-page-xl.com'
import ssl


# This restores the same behavior as before.


@st.cache_resource
def loadLLMmodel(input_url):
    req = Request(input_url)
    context = ssl._create_unverified_context()
    html_page = urlopen(req, context=context)

    soup = BeautifulSoup(html_page, "lxml")

    links = []
    for link in soup.findAll('a'):
        absoluteLink = urllib.parse.urljoin(input_url, link.get('href'))
        if "persistent.com" not in absoluteLink:
            continue
        links.append(absoluteLink)

    print("Total links found" + str(len(links)) + ", Links Found" + str(links))
    print("Using first 10 links found as data")
    loader = WebBaseLoader(links[:10])
    # ["https://labour.gov.in/sites/default/files/code_on_wages_central_advisory_board_rules2021.pdf",
    #  "https://labour.gov.in/sites/default/files/ir_gazette_of_india.pdf",
    #  "https://labour.gov.in/sites/default/files/ss_code_gazette.pdf",
    #  "https://labour.gov.in/sites/default/files/the_code_on_wages_2019_no._29_of_2019.pdf",
    #  "https://blog.ipleaders.in/labour-laws/"])
    print("Loading data now")
    data = loader.load()

    # %%
    print("data loading complete, Creating splits")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    print("Splits created, Creating Vector Store DB")
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())
    print("Creating retriever object")
    retriever = vectorstore.as_retriever()
    print("Creating llm object")
    llm = GPT4All(

        model="C:/Users/saura/PycharmProjects/pythonProject/ggml-model-gpt4all-falcon-q4_0.bin",
        max_tokens=4048,

    )
    # %%
    # question = "What are the approaches to Task Decomposition?"
    # question = promptArg
    # docs = vectorstore.similarity_search(question)
    # len(docs)
    # print(docs[0])
    # Prompt
    # template = """You are a chatbot having a conversation with a human.
    # Use the following pieces of context to answer the question at the end.
    # If you don't know the answer, just say that you don't know, don't try to make up an answer.
    # Use three sentences maximum and keep the answer as concise as possible.
    # Always say 'thanks for asking!' at the end of the answer.
    # Given the following extracted parts of a long document and a question, create a final answer.
    #
    # {context}
    #
    # {chat_history}
    # Human: {human_input}
    # Chatbot:"""

    # prompt = PromptTemplate(
    #     input_variables=["chat_history", "human_input", "context"], template=template
    # )
    print("Creating memory object")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # qa_chain = load_qa_chain(
    #     llm=llm, chain_type="stuff", memory=memory, prompt=prompt
    # )
    # template = """Use the following pieces of context to answer the question at the end.
    #         If you don't know the answer, just say that you don't know, don't try to make up an answer.
    #         Use three sentences maximum and keep the answer as concise as possible.
    #         Always say "thanks for asking!" at the end of the answer.
    #         """

    # memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)

    # %%

    # QA_CHAIN_PROMPT = PromptTemplate(
    #   input_variables=["context", "question"],
    #   template=template,
    # )

    # Chain
    # chain = load_qa_chain(llm, chain_type="stuff", prompt=QA_CHAIN_PROMPT)

    # Run
    # print(chain({"input_documents": docs, "question": question}, return_only_outputs=True))
    print("Creating qa_chain object")
    # Define the system message template
    # system_template = """Only Use the following pieces of context and chat history to answer the question at the end.
    #                     If you don't know the answer, just say that you don't know, don't try to make up an answer.
    #                     If answer generated is not relevant to context, just say question is not relevant.
    #                     Given the following extracted parts of a long document and a question, create a final answer.
    #                     Combine the chat history and follow up question into a standalone question.
    #                     Context: {context}
    #                     Chat History:
    #                     {chat_history}
    #                     Follow Up Input: {question}
    #                     Standalone question:"""
    #
    # PROMPT = PromptTemplate(
    #     input_variables=["chat_history", "question", "context"],
    #     template=system_template
    # )

    # Create the chat prompt templates
    # messages = [
    #     SystemMessagePromptTemplate.from_template(system_template),
    #     HumanMessagePromptTemplate.from_template("{question}")
    #
    # ]
    # qa_prompt = ChatPromptTemplate.from_messages(messages)

    # %%
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True
    )

    # qa_chain = RetrievalQA.from_chain_type(
    #   llm,
    #   retriever=vectorstore.as_retriever(),
    #   chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    # )
    print("Returning QA Chain object ")
    return qa_chain


st.title("Pi ChatBot ")


def conversation_chat(query):
    qa_chainObject = loadLLMmodel("https://www.persistent.com")
    result = qa_chainObject({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    print("History " + str(st.session_state['history']))
    return result["answer"]


def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey!"]


def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask me about labour law codes.", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversation_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            print(st.session_state['generated'])

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")


# Initialize session state
initialize_session_state()
# Display chat history
display_chat_history()
