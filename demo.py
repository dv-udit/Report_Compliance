from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from datetime import datetime

import time
import subprocess
import os
import streamlit as st
import sys
__import__('pysqlite3')

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

st.title('🦜🔗 Report Compliance')

if "file_name" not in st.session_state:
    st.session_state.file_name = ""


def document_loader():
    name = st.session_state.file_name
    file_path = f"test/{name}"
    #   directory = os.getcwd()  # Get the current working directory

    loader = UnstructuredExcelLoader(file_path, mode="paged")
    documents = loader.load()

    return documents


def split_documents(docs):
    for doc in docs:
        doc.metadata["languages"] = "eng"
        doc.page_content = doc.page_content.replace("\n", " | ")

    return docs


def get_embedding():
    embedding_function = OpenAIEmbeddings()
    return embedding_function


def set_db(attributes):
    documents = document_loader()
    docs = split_documents(documents)
    embeddings = get_embedding()

    db = Chroma().from_documents(documents=docs, embedding=embeddings,
                                  collection_metadata={"hnsw:space": "cosine"})
    pages = set([doc.metadata["page_name"] for doc in documents])

    prompt = PromptTemplate(template="""

{question}

Question: Check if the following word is present using following logic by displaying your chain of thought as well:
Thought 1 I need to search if the word or its full form is present in the document text as it is first.
Observation 1 (Result 1/2) The [Word] is directly present, I can now finish and go to Action 5.
Observation 1 (Result 2/2) The [Word] is not directly present, I need to search for full form now.
Action 1 Fuzzy Match[Word, Document text]
Thought 2 I need to verify if the full form of the word is present in the document text.
Action 2 Fuzzy Match[Word Full Form, Document text]
Observation 2 (Result 1/2) The [Word Full Form] is directly present, I can now finish and go to Action 5.
Observation 2 (Result 2/2) The [Word Full Form] is not directly present, I need to do semantic search now.
Thought 3 Full form is not present, I need to verify if there is similar phrase that has same semantic meaning as the word.
Action 3 Semantic Search[Word Description, Document text]
Observation 3 (Result 1/2) The phrase having similar meaning is not present, I cannot find a match, I can now go to Action 5.
Observation 3 (Result 2/2) The phrase having similar meaning is present.
Thought 4 I need to verify if the phrase has same semantic meaning as the word by understanding the definition.
Action 4 Verify Same[Word Description, Phrase]
Observation 5 (Result 1 / 1) The phase and word description are related and have same meaning so display Finish[Match] and exit.
Observation 5 (Result 1 / 2) The phase and word description are related but do not have same meaning, I can now go to Action 5.
Action 5 Finish[No match]

Document text is specified between the quotes:
```
{context}
```
""", input_variables=["context", "question"])
    
    for page_name in pages:
    # for page_name in ["LIQB"]:
        retriever = db.as_retriever(search_type="mmr",
                                    search_kwargs={'filter': {
                                        'page_name': page_name}, 'k': 1}
                                    )
        llm = OpenAI(temperature=0,model_name = 'gpt-4')  # model_name="text-davinci-003"
        chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=retriever,
                                    chain_type_kwargs={"prompt": prompt},
                                    return_source_documents=True)
        st.write("------------------------------------------------------------------")
        st.write("Page: " + str(page_name))
        st.write("------------------------------------------------------------------")

        for query in attributes:
            attribute_name = query['attribute_name']
            attribute_description = query['attribute_description']
            query = f"Word: {attribute_name} \nDescription: {attribute_description}"
            response = chain({'query': query})
            # print(response)
            output = {}
            output = {"attribute" : response['query'] , "response" : response['result']}
        
            st.write("Attribute : \n",output["attribute"],"\n")
            st.write("Response : \n",output["response"],"\n")
        

def save_uploaded_file(uploadedfile):
    file_name = uploadedfile.name
    current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    final_filename = current_datetime + file_name
    # st.write(final_filename)
    st.session_state.file_name = final_filename
    with open(os.path.join("test", final_filename), "wb") as f:
        f.write(uploadedfile.getbuffer())
    # return st.success("Saved file :{} in tempDir".format(uploadedfile.name))
    return st.success("Saved file :{} in tempDir".format(final_filename))


def file_upload_form():
    with st.form('fileform'):
        supported_file_types = ["xlsx"]
        uploaded_file = st.file_uploader(
            "Step 1: Upload Your Report file (xlsx)", type=(supported_file_types))
        # st.write(uploaded_file)
        submitted = st.form_submit_button("Upload")
        # st.write(uploaded_file.path)
        if submitted:
            if uploaded_file is not None:
                if uploaded_file.name.split(".")[-1] in supported_file_types:
                    st.write("File Uploaded successfully")

                    file_details = {
                        "FileName": uploaded_file.name,
                        "FileType": uploaded_file.type
                                    }
                    st.session_state.current_filename = uploaded_file.name ,

                    save_directory = "test"
                    os.makedirs(save_directory, exist_ok=True)
                    # st.write("Directory created successfully")

                    save_uploaded_file(uploaded_file)

                    # Process and save the uploaded file to the desired location
                    # process_and_save_file(uploaded_file, save_directory)

                else:
                    st.write(
                        f"Supported file types are {', '.join(supported_file_types)}")
            else:
                st.write("Please select a file to upload first!")
                # with st.spinner('Generating...'):
                #     generate_response(query_text, filename)


def query_form():
    with st.form('myform'):
        query_text = st.text_area(
            'Step 2: Catalog:', placeholder='Catalog contents')
        submitted = st.form_submit_button(
            'Submit')
        if submitted:
            with st.spinner('Generating...'):
                # text_array = query_text.split('\n')
                # print(type(text_array))
                text_array = query_text.strip().split('\n')
                attributes_dict = []
                for i in text_array:
                    attribute_name, description = i.split(',', 1)
                    attribute_name = f'{attribute_name.strip()}'
                    description = f'{description.strip()}'
                    dict = {"attribute_name" : attribute_name, "attribute_description" : description}
                    attributes_dict.append(dict)
                    
                # for i in attributes_dict:
                #     print(i)
                set_db(attributes_dict)

file_upload_form()
query_form()
