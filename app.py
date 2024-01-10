from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain import OpenAI
from langchain.chains import RetrievalQA
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


def split_documents(documents):

    # split it into chunks
    text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    # docs = text_splitter.create_documents(text)

    # Fix: was throwing error before
    # metadata only supports primitive types such as str, int, etc
    for doc in docs:
        doc.metadata["languages"] = "eng"

    return docs


def get_embedding():
    embedding_function = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2")

    return embedding_function


def set_db(attributes):
    documents = document_loader()
    docs = split_documents(documents)
    embeddings = get_embedding()

    db = Chroma().from_documents(documents=docs, embedding=embeddings,
                                 persist_directory="/tmp/brcl_03", collection_metadata={"hnsw:space": "cosine"})
    places = set([doc.metadata["page_name"] for doc in documents])
    # places = set({'LIQB', 'OV1', 'CR8', 'CONTENTS', 'Catalog', 'CCR7', 'MR2-B', 'LIQ1', 'KM1'})
    # places.remove("Catalog")

    prompt = PromptTemplate(template="""
    You are an intelligent bot, who does not hallucinate. Answer in following format:
    [
    expected_words: [{question}],
    answer: <yes or no by checking if fuzzy match of expected_words is present in file, assuming that we do not want similar or related words, we only want ones which matching {question}>,
    reasoning: <Verify and give reason you think {question} are not exact fuzzy match with the file, assuming that we do not want similar or related words or concepts, we want only ones which match {question}>,
    percentage_match: <x% based of fuzzy match with {question}>
    ]

    ---
    File: 
    {context}
    ---
""", input_variables=["context"])

    for page_name in places:
        # print("\n----------------------------------------------------\n")
        # print(f"\nPage Name- {page_name}\n")
        st.write(f"----------------------")
        st.write(f" Page Name- {page_name}")
        st.write(f"----------------------")
    # print("\n----------------------------------------------------\n")

        # attributes = ["CCR or Counterparty credit risk", "                                                                                                                                                       ", "IRC or Incremental Risk Charge", "A-IRB or Advance internal ratings-based Approach",
        #               "SREP or Supervisory Review and Evaluation Process", "TURF or Total Unduplicated Reach and Frequency", "LCR or Liquidity Coverage Ratio", "HLBA or historical look-back approach", "RWEA"]

        for query in attributes:
            retriever = db.as_retriever(search_type="mmr",
                                        search_kwargs={'filter': {
                                            'page_name': page_name}, 'k': 3}
                                        )
            llm = OpenAI(temperature=0)  # model_name="text-davinci-003"
            chain = RetrievalQA.from_chain_type(llm=llm,
                                                chain_type="stuff",
                                                retriever=retriever,
                                                chain_type_kwargs={"prompt": prompt})

            response = chain({'query': query})
            # print(f"{response['query']}")
            # print(f"{response['result']}")
            st.write(f"{response['query']}")
            st.write(f"{response['result']}")
            # print("\n\n\n\n")


def save_uploaded_file(uploadedfile):
    file_name = uploadedfile.name
    st.session_state.file_name = file_name
    with open(os.path.join("test", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved file :{} in tempDir".format(uploadedfile.name))


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
                    # set_LLM(uploaded_file)
                    # st.session_state.current_filename = uploaded_file.name
                    st.write("File Uploaded successfully")

                    file_details = {"FileName": uploaded_file.name,
                                    "FileType": uploaded_file.type}

                    save_directory = "test"
                    os.makedirs(save_directory, exist_ok=True)
                    # st.write("Directory created successfully")

                    save_uploaded_file(uploaded_file)

                    # Process and save the uploaded file to the desired location
                    # process_and_save_file(uploaded_file, save_directory)

                    # set_db()
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
                # generate_response(query_text, filename)
                # file_qa = st.session_state.QA[filename]
                # for i in file_qa:
                #     st.write("Question : " + i["question"])
                #     st.write("Answer : " + i["answer"])
                # lines_array = query_text.splitlines()
                text_array = query_text.split('\n')

                set_db(text_array)

                # st.write(type(lines_array))
                # st.write(type(query_text))


file_upload_form()
query_form()
