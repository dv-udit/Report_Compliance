from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import time
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

# from attributes import phrases, prompt

import os



embedding_function = OpenAIEmbeddings()

# # load the document and split it into chunks
# file_path = "Risk Report - Test Data LIQB.xlsx"
# file_path = "Risk Report - Test Data LIQB (1).xlsx"
file_path = "Risk Report - Test Data 3 (1).xlsx"

loader = UnstructuredExcelLoader(file_path, mode="paged")
documents = loader.load()

# # metadata only supports primitive types such as str, int, etc
for doc in documents:
  doc.metadata["languages"] = "eng"
  doc.page_content = doc.page_content.replace("\n", " | ")



# load it into Chroma
db = Chroma().from_documents(documents=documents, embedding=embedding_function, collection_metadata={"hnsw:space": "cosine"})
pages = set([doc.metadata["page_name"] for doc in documents])


print(pages)
# pages.remove("Catalog")
# pages.remove("CONTENTS")


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
Action 4 Verify Same[Word, Phrase]
Observation 5 (Result 1 / 1) The phase and words are related but do not have same meaning.
Action 5 Finish[No match]

Document text is specified between the quotes:
```
{context}
```
  """, input_variables=["context", "question"])

# attributes = ["'A-IRB' or 'Advance internal ratings-based Approach'", "'SREP' or 'Supervisory Review and Evaluation Process'", "'TURF' or 'Total Unduplicated Reach and Frequency'", "'LCR' or 'Liquidity Coverage Ratio'", "'HLBA' or 'historical look-back approach'", "'RWEA' or 'Risk-Weighted Exposure Amount'"]



for page in pages:
  retriever = db.as_retriever(
      search_kwargs={'filter': {'page_name': page}, 'k': 5}
  )
  llm = OpenAI(model_name="gpt-4")  # model_name="text-davinci-003"
  chain = RetrievalQA.from_chain_type(llm=llm,
                                      chain_type="stuff",
                                      retriever=retriever,
                                      chain_type_kwargs={"prompt": prompt},
                                      return_source_documents=True)

  print("------------------------------------------------------------------")
  print("Page: " + str(page))
  print("------------------------------------------------------------------")

  phrase = '''
  Word: Dividend Yield
  Description: Income Efficiency - A financial ratio measuring the percentage return on investment through dividends, indicating the profit-sharing efficiency of a stock or portfolio.

  '''

  

  response = chain(phrase)
  print(f"{response['result']}")
  print("\n\n\n\n")
  print("\n\n\n\n")

