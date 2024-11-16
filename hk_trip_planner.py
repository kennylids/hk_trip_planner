import streamlit as st
import os

if st.secrets["IS_PRODUCTION"]=='True':

    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

    os.environ['LANGCHAIN_TRACING_V2'] = st.secrets["LANGCHAIN_TRACING_V2_"]
    os.environ['LANGCHAIN_ENDPOINT'] = st.secrets["LANGCHAIN_ENDPOINT_"]
    os.environ['LANGCHAIN_API_KEY'] = st.secrets["LANGCHAIN_API_KEY_"]
    os.environ['LANGCHAIN_PROJECT'] = st.secrets["LANGCHAIN_PROJECT_"]

os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["hf_access_token"]



from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma

# from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
# from langchain_huggingface import HuggingFaceEmbeddings

from InstructorEmbedding import INSTRUCTOR

# from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings

from operator import itemgetter

import re


from langchain.prompts import ChatPromptTemplate


# 1. vector-store pre-rendered
persist_directory = 'large_db'
model_name = "hkunlp/instructor-large"
# persist_directory = 'base_db'
# model_name = "hkunlp/instructor-base"

# Supplying a persist_directory will store the embeddings on disk
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name=model_name, 
                                                      model_kwargs={"device": "cpu"})
embedding = instructor_embeddings

# Now we can load the persisted database from disk, and use it as normal. 
vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding)

# retriever = vectordb.as_retriever()
retriever = vectordb.as_retriever(search_kwargs={"k": 3})


#connect huggingface llm llama
llm = HuggingFaceEndpoint(
    #  repo_id="microsoft/Phi-3-mini-4k-instruct",
    # repo_id="microsoft/Phi-3.5-mini-instruct",
     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    ####repo_id="meta-llama/Llama-3.2-3B-Instruct",

    task="text-generation",
    # max_new_tokens=4095,
    temperature=0.01,
    do_sample=False
)


template_travel = """You are a very helpful travel guide in Hong Kong.  Your goal is to answer each question, using the following documents as context, as truthfully as you can. No need to tell the users that you are referencing contexts.:
{context}

Question: {question}
Travel guide:
"""
prompt = ChatPromptTemplate.from_template(template_travel)

# user_input = "I want a plan for 5-day vacation in Hong Kong. I want an iterinary where I can have good siu mei and dim sum and then some coffee. I want to spend one day hiking exploring nature as well."
# question = user_input

# RAG-Fusion: Related
template = """You are a helpful traveling assistant in Hong Kong that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (3 queries) which each is started with ***:"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)


generate_queries = (
    prompt_rag_fusion 
    | llm
    | StrOutputParser() 
    # | (lambda x: x.split("***")[1:4])
    | (lambda x: re.findall(r'\*\*\*(.*?)\*\*\*', x))
    
)

from langchain.load import dumps, loads

def reciprocal_rank_fusion(results: list[list], k=60,get_nth_results=6):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results[:get_nth_results]

retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion

final_rag_chain = (
    {"context": retrieval_chain_rag_fusion, 
     "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)




# Title
st.markdown("<h1 style='text-align: center; font-size: calc(1.5rem + 1vw);'>Hong Kong 🔥 🇭🇰 🔥 Trip Planner</h1>", unsafe_allow_html=True)

def generate_response(question):
    answer = final_rag_chain.invoke({"question":question})
    st.info(answer)

# Main content
st.markdown("<h4 style='text-align: center; color: #FF5733; font-size: calc(1rem + 0.5vw);'>How do you want to plan your trip in Hong Kong?</h4>", unsafe_allow_html=True)

# Form
with st.form("my_form"):
    st.markdown("<h4 style='text-align: center; font-size: calc(1rem + 0.5vw);'>ASK AWAY!!!</h4>", unsafe_allow_html=True)
    text = st.text_area(
        'Below is an example. I encourage you to think about what kind of things you\'d like to see or do in the city. Coffee, dumplings, Michelin foods, nature, hikes, culture, luxury stuff, weather, shopping....',
        "I want a 5-day vacation plan in Hong Kong. I want an itinerary where I can have good siu mei and dim sum and then some coffee. I want to spend one day hiking exploring nature as well. ",
        height=150  # Fixed height for better mobile display
    )
    
    col1, col2 = st.columns([6, 1])  # Create two columns
    with col2:
        submitted = st.form_submit_button("Submit")

    if submitted:
        with st.spinner("Querying embeddings & Inferencing..."):
            generate_response(text)  # Call your function to generate a response

if submitted:
    with st.spinner("Querying embeddings & Inferencing..."):
        generate_response(text)

# Footer with improved mobile responsiveness
st.markdown("""
<style>
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 10px;
    font-size: 0.8rem;
    border-top: 1px solid #ccc;
    max-height: 25vh;
    overflow-y: auto;
}
.footer-content {
    max-width: 800px;
    margin: 0 auto;
    text-align: center;
}
@media (max-width: 768px) {
    .footer {
        position: relative;
        margin-top: 2rem;
        font-size: 0.7rem;
        padding: 8px;
    }
}
</style>
<div class="footer">
    <div class="footer-content">
        This application utilizes advanced AI technologies, including Hugging Face model (Meta Llama-3-8B), Chroma embeddings, and LangChain framework to provide travel recommendations in Hong Kong.<br>
        Huge thanks to HKU NLP Department for opensourcing their instructor-large text embedding model for making this computationally easy.<br>
        There is slight chance of the AI hallucinating, so please forget and forgive if the AI makes up a non-existent place or district 😛<br>
        Inquiry: kenchinsonxyz@gmail.com
    </div>
</div>
""", unsafe_allow_html=True)