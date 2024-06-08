import os
from typing import List
from time import sleep
from logging import info
import openai
from dotenv import load_dotenv
import requests
from nltk.tokenize import word_tokenize
import tiktoken
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

OPEN_AI_CLIENT = os.getenv("OPENAI_API_KEY")
OPEN_AI_EMBEDDINGS = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

OPENAI_BACKOFF_SECONDS = 20  # 3 requests per minute
OPENAI_MAX_RETRIES = 3
OLLAMA_API_ENDPOINT = os.getenv("OLLAMA_URL")
AI_SYSTEM_MESSAGE = {
        "role": "system",
        "content": """You're a helpful AI Code Reviewer."""
    }
MESSAGES = [AI_SYSTEM_MESSAGE]
VECTOR_STORE = None
PROJECT_README_CONTENT = ""


def code_type(filename: str) -> str:
    """
    Determine the programming language of a file based on its extension.
    """
    extension = filename.split(".")[-1].lower()
    if "js" in extension:
        return "JavaScript"
    elif "ts" in extension:
        return "TypeScript"
    elif "java" in extension:
        return "Java"
    elif "py" in extension:
        return "Python"
    else:
        return extension.replace(".", "").upper()

def prompt(filename: str, contents: str) -> str:
    """
    Generate a code review prompt for a specific file.
    """
    code = "code"
    type = code_type(filename)
    if type:
        code = f"{type} {code}"

    return (
        f"Please perform a code review (keep the feedback within 600 words) on the {code} specifically named {filename} below inside the triple backticks.\n"
        f"File contents:\n```\n{contents}\n```"
        "Use must use the following checklist inside the triple backticks below to guide your analysis and review accordingly:\n"
        "```"
        "   1. Documentation Defects:\n"
        "       a. Naming: Assess the quality of software element names.\n"
        "       b. Comment: Analyze the quality and accuracy of code comments.\n"
        "   2. Visual Representation Defects:\n"
        "       a. Bracket Usage: Identify any issues with incorrect or missing brackets.\n"
        "       b. Indentation: Check for incorrect indentation that affects readability.\n"
        "       c. Long Line: Point out any long code statements that hinder readability.\n"
        "   3. Structure Defects:\n"
        "       a. Dead Code: Find any code statements that serve no meaningful purpose.\n"
        "       b. Duplication: Identify duplicate code statements that can be refactored.\n"
        "   4. New Functionality:\n"
        "       a. Use Standard Method: Determine if a standardized approach should be used for single-purpose code statements.\n"
        "   5. Resource Defects:\n"
        "       a. Variable Initialization: Identify variables that are uninitialized or incorrectly initialized.\n"
        "       b. Memory Management: Evaluate the program's memory usage and management.\n"
        "   6. Check Defects:\n"
        "       a. Check User Input: Analyze the validity of user input and its handling.\n"
        "   7. Interface Defects:\n"
        "       a. Parameter: Detect incorrect or missing parameters when calling functions or libraries.\n"
        "   8. Logic Defects:\n"
        "       a. Compute: Identify incorrect logic during system execution.\n"
        "       b. Performance: Evaluate the efficiency of the algorithm used.\n"
        "Provide your feedback in a numbered list for each category. At the end of your answer, summarize the recommended changes to improve the quality of the code provided.\n"
        "```"
    )

def review(
    filename: str, content: str, model: str, temperature: float, 
    max_tokens: int, file_contents: List[str]) -> str:
    """
    Review a code snippet using the specified model.
    """
    global VECTOR_STORE
    rag_mode = file_contents is not None and len(file_contents) > 0
    if "gpt" in model.lower():
        VECTOR_STORE = Chroma.from_texts(texts=get_split_text(file_contents),
                                        embedding=OPEN_AI_EMBEDDINGS) if rag_mode and VECTOR_STORE is None else VECTOR_STORE
        return review_with_openai(filename, content, model, temperature, max_tokens, VECTOR_STORE, rag_mode)
    else:
        VECTOR_STORE = Chroma.from_texts(texts=get_split_text(file_contents),
                                        embedding=OllamaEmbeddings(model=model)) if rag_mode and VECTOR_STORE is None else VECTOR_STORE
        return review_with_ollama(filename, content, model, temperature, max_tokens, VECTOR_STORE, rag_mode)

def review_with_ollama(filename: str, content: str, model: str,
                       temperature: float, max_tokens: int,
                       vectorstore: Chroma, rag_mode: bool) -> str:
    """
    Review a code snippet using the OLLAMA API.
    """
    global MESSAGES
    try:
        # print(f"Context for {filename}: \n")
        # print('Content length: ', len(context), 'Context Content:', context)
        # print('\n\n\n')
        if(rag_mode):
            context = get_relevant_rag_context(prompt(filename, content), vectorstore)
            MESSAGES.append({"role": "user", "content": f"{prompt(filename, content)},\n"
                + f"Furthermore, use the following context inside the triple backticks to answer the above question:\nContext: '''\n{context}\n'''"})
        else:
            MESSAGES.append({"role": "user", "content": prompt(filename, content)})
        # Reset messages if we exceed max tokens
        reset_messages_if_exceeds_max_tokens(filename, content, model, max_tokens)
        data = {
            "model": model,
            "messages": MESSAGES,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(OLLAMA_API_ENDPOINT, json=data, headers=headers, timeout=100)
        chat_review = response.json()['message']['content']
        # print(f"Response for {filename}: \n")
        # print(chat_review)
        # print('\n\n\n')
        MESSAGES.append({"role": "assistant", "content": chat_review})
        return f"{model.capitalize()} review for {filename}:*\n" f"{chat_review}"
    except Exception as e:
        print(f'Failed to review file {filename}: {e}')

def review_with_openai(
    filename: str, content: str, model: str, temperature: float, 
    max_tokens: int, vectorstore: Chroma, rag_mode: bool) -> str:
    """
    Review a code snippet using the OpenAI API.
    """
    x = 0
    global MESSAGES
    while True:
        try:
            if(rag_mode):
                context = get_relevant_rag_context(prompt(filename, content), vectorstore)
                MESSAGES.append({"role": "user", "content": f"{prompt(filename, content)},\n"
                    + f"Furthermore, use the following context inside the triple backticks to answer the above question:\nContext: '''\n{context}\n'''"})
            else:
                MESSAGES.append({"role": "user", "content": prompt(filename, content)})    
            # Reset messages if we exceed max tokens
            reset_messages_if_exceeds_max_tokens(filename, content, model, max_tokens)
            chat_review = (
                OPEN_AI_CLIENT.chat.completions.create(model=model,
                                                       
                temperature=temperature,
                max_tokens=max_tokens,
                messages=MESSAGES)
                .choices[0]
                .message.content
            )
            # print(chat_review)
            # print('\n\n\n')
            MESSAGES.append({"role": "assistant", "content": chat_review})
            return f"{model.capitalize()} review for {filename}:*\n" f"{chat_review}"
        except openai.RateLimitError:
            if x < OPENAI_MAX_RETRIES:
                info("OpenAI rate limit hit, backing off and trying again...")
                sleep(OPENAI_BACKOFF_SECONDS)
                x+=1
            else:
                raise Exception(
                    f"""finally failing request to OpenAI platform for code review,
                    max retries {OPENAI_MAX_RETRIES} exceeded"""
                )
            
def reset_messages_if_exceeds_max_tokens(filename: str, content: str, model: str, max_tokens: int):
    """
    Reset messages if the token length exceeds the maximum tokens allowed.
    """
    global MESSAGES
    concatenated_messages = "".join([message["content"] for message in MESSAGES])
    current_token_length = get_token_length_in_words(concatenated_messages, model)
    print(f"{filename}, Token length: " + str(current_token_length))
    if(current_token_length > max_tokens):
        print(f'Resetting messages as tokens are greater than max tokens at {filename}')
        MESSAGES = [AI_SYSTEM_MESSAGE]
        MESSAGES.append({"role": "user", "content": prompt(filename, content)})


def get_token_length_in_words(text: str, model: str) -> int:
    """
    Get the token length of a text based on the model used.
    """
    if("gpt" in model.lower()):
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    else:
        tokens = word_tokenize(text)
        return len(tokens)


def load_docs_from_vector_store(query, embeddings):
    """
    Load documents from the vector store.
    """
    db3 = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    docs =  db3.similarity_search(query)
    return docs

def get_split_text(file_contents):
    """
    Split the text into chunks.
    """
    splits = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=200)
    if PROJECT_README_CONTENT != "":
        file_contents.append(PROJECT_README_CONTENT)
    for file_content in file_contents:
        splits = splits + text_splitter.split_text(file_content)
    return splits
def combine_docs(docs):
    """
    Combine the documents into a single string.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def get_relevant_rag_context(question, vectorstore):
    """
    Get the relevant RAG context.
    """
    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.invoke(question)
    print(f"Retrieved {len(retrieved_docs)} docs from vectorstore")
    # print('\n' + combine_docs(retrieved_docs))
    return combine_docs(retrieved_docs)

def start_console_chat_with_llm():
    """
    Chat with the language model in the terminal.
    """
    global MESSAGES
    model = os.getenv("AI_MODEL")
    print("Type 'exit' to quit the chat.")
    while True:
        user_input = input("Please enter your question to the AI here: ")
        if(user_input == "exit"):
            break
        MESSAGES.append({"role": "user", "content": user_input})
        if("gpt" in model.lower()):
            response = (
                OPEN_AI_CLIENT.chat.completions.create(
                    model=model,
                    messages=MESSAGES,
                    max_tokens=int(os.getenv("AI_MAX_TOKENS")),
                    temperature=0.5,
                )
                .choices[0]
                .message.content
            )
        else:
            response = (
                requests.post(
                    OLLAMA_API_ENDPOINT,
                    json={
                        "model": model,
                        "messages": MESSAGES,
                        "temperature": 0.1,
                        "max_tokens":int(os.getenv("AI_MAX_TOKENS")),
                        "stream": False,
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=30000,
                )
                .json()["message"]["content"]
            )
        print("\n\n")
        print(f"AI: {response}")
        MESSAGES.append({"role": "assistant", "content": response})
