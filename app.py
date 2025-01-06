import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,YoutubeAudioLoader,UnstructuredURLLoader

from langchain.schema import Document
from youtube_transcript_api import YouTubeTranscriptApi

from langchain.text_splitter import RecursiveCharacterTextSplitter


## streamlit app
st.set_page_config(page_title="Langchain: Summarize text from Youtube or Website", page_icon="🦜🔗")
st.title("🦜🔗 Langchain: Summarize text from Youtube or Website")
st.subheader('Summarize URL')

## Get the Groq API key and url to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

url = st.text_input("URL",label_visibility="collapsed")



## LLM model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it", streaming=True)


chunk_template = """
Provide a summary of the following content:
Speech : {text}
"""
prompt = PromptTemplate(template=chunk_template, input_variables=["text"])


final_prompt = """
Provide the final summary of the entire speech with these important points
Add a title, start the precise summary with an introduction and provide the summary in number points for the speech.
Speech : {text}
"""
final_prompt_template = PromptTemplate(input_variables=['text'],template=final_prompt)


if st.button("Summarize the Contenct from YT or Website"):
    ## Validate the inputs
    if not groq_api_key.strip() or not url.strip():
        st.error("Please provide the information")
    
    elif not validators.url(url):
        st.error("Please enter a valid URL")
    
    else:
        try:
            with st.spinner("Waiting..."):
                ## loading the data
                if "youtube.com" in url:
                    v_id = url.split("v=")[-1]
                    transcript = YouTubeTranscriptApi.get_transcript(video_id=v_id)
                    text = " ".join([entry['text'] for entry in transcript])
                    docs = [Document(page_content=text)]
                    # loader = YoutubeLoader.from_youtube_url(url,add_video_info=True)
                else:
                    loader= UnstructuredURLLoader(urls=[url],ssl_verify=False,
                                                  headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"}
                                                  )
                    docs = loader.load()

                
                final_docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)

                ## chain for summarization
                summary_chain = load_summarize_chain(
                                                     llm = llm,
                                                     chain_type="map_reduce",
                                                     map_prompt = prompt,
                                                     combine_prompt=final_prompt_template,
                                                     verbose=True)
                output_summary= summary_chain.run(final_docs)
                st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception: {e}")