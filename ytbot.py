import gradio as gr
import re
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# Video ID extraction function
'''
def get_video_id(url):
    # Regular expression to match YouTube video URLs
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern,url)
    return match.group(1) if match else None
'''
def get_video_id(url):
    """Extract YouTube video ID from common YouTube URL formats."""

    parsed_url = urlparse(url)

    # https://www.youtube.com/watch?v=VIDEO_ID
    if parsed_url.hostname in ["www.youtube.com", "youtube.com", "m.youtube.com"]:
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query).get("v", [None])[0]

        # https://youtube.com/shorts/VIDEO_ID
        if parsed_url.path.startswith("/shorts/"):
            return parsed_url.path.split("/")[2]

        # https://youtube.com/embed/VIDEO_ID
        if parsed_url.path.startswith("/embed/"):
            return parsed_url.path.split("/")[2]

    # https://youtu.be/VIDEO_ID
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")

    return None

# Fetching transcript from youtube
def get_transcript(url):
    # Extract the video ID from the URL
    video_id = get_video_id(url)

    # Create a YouTubeTranscriptApi() object
    ytt_api = YouTubeTranscriptApi()

    # Fetch the list of available transcripts for the given youtube video
    transcripts = ytt_api.list(video_id)

    transcript = ""
    for t in transcripts:
        # Check if the trnscript's language is English
        if t.language_code == "en":
            if t.is_generated:
                # If no transcript has been set yet, use the auto generated one
                if len(transcript) == 0:
                    transcript = t.fetch()

            else:
                # if a manually created transcript is found, use it (overrides auto generated)
                transcript = t.fetch()
                break

    return transcript if transcript else None

# Processing the Transcript
def process(transcript):

    txt =  ""
    for i in transcript:
        try:
            txt += f"Text: {i.text} Start: {i.start}\n"
        except KeyError:
            pass
    return txt


# Chunking the transcript
def chunk_transcript(processed_transcript,chunk_size=200,chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap = chunk_overlap
    )

    chunks = text_splitter.split_text(processed_transcript)
    return chunks

# setting up the model and embeddings
model = ChatAnthropic(model = "claude-haiku-4-5-20251001")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Similarity search(FAISS) setup
def create_faiss_index(chunks,embedding_model):
    """
    Create a FAISS index from text chunks using the specified embedding model.

    :param chunks: List of text chunks
    :param embedding_model: The embedding model to use
    <span data-type="emoji" data-name="return"></span> FAISS index
    """

    return FAISS.from_texts(chunks, embedding_model)

# Performing similarity search
def perform_similarity_search(faiss_index, query, k=3):
    """
    Search for specific queries within the embedded transcript using the FAISS index.
    
    :param faiss_index: The FAISS index containing embedded text chunks
    :param query: The text input for the similarity search
    :param k: The number of similar results to return (default is 3)
    <span data-type="emoji" data-name="return"></span> List of similar results
    """

    results = faiss_index.similarity_search(query, k=k)
    return results

# Summarizing the transcript
def create_summary_prompt():
    """
    Create a prompt template for summarizing the transcript.

    <span data-type="emoji" data-name="return"></span> PromptTemplate object
    """

    summary_template = """
    You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.
    Instructions:
    1. Summarize the transcript in a single concise paragraph.
    2. Ignore any timestamps in your summary.
    3. Focus on the spoken content (Text) of the video.

    Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Please summarize the following YouTube video transcript:
    
    Transcript:
    {transcript}
    
    Summary:
    """

    prompt = PromptTemplate(
        input_variables=["transcript"],
        template=summary_template
    )
    return prompt

# LLMChain for summarization
def create_summary_chain(llm, prompt, verbose=True):
    """
    Create an LLMChain for generating summaries.
    
    :param llm: Language model instance
    :param prompt: PromptTemplate instance
    :param verbose: Boolean to enable verbose output (default: True)
    <span data-type="emoji" data-name="return"></span> LLMChain instance
    """
    return LLMChain(llm=llm, prompt=prompt, verbose=verbose)

# Retrieving relevant context
def retrieve(query, faiss_index, k=7):
    """
    Retrieve relevant context from the FAISS index based on the user's query.

    Parameters:
        query (str): The user's query string.
        faiss_index (FAISS): The FAISS index containing the embedded documents.
        k (int, optional): The number of most relevant documents to retrieve (default is 3).

    Returns:
        list: A list of the k most relevant documents (or document chunks).
    """
    relevant_context = faiss_index.similarity_search(query, k=k)
    return relevant_context

# Creating the Q&A prompt template
def create_qa_prompt_template():
    """"
    Create a PromptTemplate for question answering based on video content.

    Returns:
        PromptTemplate: A PromptTemplate object configured for Q&A tasks.
    """

    # Define the template string
    qa_template = """
    You are an expert assistant providing detailed answers based on the following video content.

    Relevant Video Context: {context}

    Based on the above context, please answer the following question:
    Question: {question}
    """

    # Create the PromptTemplate Object
    prompt_template = PromptTemplate(
        input_variables=["context","question"],
        template=qa_template
    )
    return prompt_template

# Setting up the Q&A chain
def create_qa_chain(llm, prompt_template, verbose=True):
    """
    Create an LLMChain for question answering.

    Args:
        llm: Language model instance
            The language model to use in the chain.
        prompt_template: PromptTemplate
            The prompt template to use for structuring inputs to the language model.
        verbose: bool, optional (default=True)
            Whether to enable verbose output for the chain.

    Returns:
        LLMChain: An instantiated LLMChain ready for question answering.
    """
    
    return LLMChain(llm=llm, prompt=prompt_template, verbose=verbose)

# Generating an Answer
def generate_answer(question, faiss_index, qa_chain, k=7):
    """
    Retrieve relevant context and generate an answer based on user input.

    Args:
        question: str
            The user's question.
        faiss_index: FAISS
            The FAISS index containing the embedded documents.
        qa_chain: LLMChain
            The question-answering chain (LLMChain) to use for generating answers.
        k: int, optional (default=3)
            The number of relevant documents to retrieve.

    Returns:
        str: The generated answer to the user's question.
    """

    # Retrieve relevant context
    relevant_context = retrieve(question, faiss_index, k=k)

    # Generate answer using the QA chain
    answer = qa_chain.predict(context=relevant_context, question=question)

    return answer


# Summarizing a video
# Initialize an empty string to store the processed transcript after fetching and preprocessing
processed_transcript = ""

def summarize_video(video_url):
    """
    Title: Summarize Video

    Description:
    This function generates a summary of the video using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.

    Returns:
        str: The generated summary of the video or a message indicating that no transcript is available.
    """
    global fetched_transcript, processed_transcript
    
    
    if video_url:
        # Fetch and preprocess transcript
        fetched_transcript = get_transcript(video_url)
        processed_transcript = process(fetched_transcript)
    else:
        return "Please provide a valid YouTube URL."

    if processed_transcript:

        llm = model

        summary_prompt = create_summary_prompt()
        summary_chain = create_summary_chain(llm, summary_prompt)

        # Step 4: Generate the video summary
        summary = summary_chain.run({"transcript": processed_transcript})
        return summary
    else:
        return "No transcript available. Please fetch the transcript first."

# Answering User's Question
def answer_question(video_url, user_question):
    """
    Title: Answer User's Question

    Description:
    This function retrieves relevant context from the FAISS index based on the user’s query 
    and generates an answer using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.
        user_question (str): The question posed by the user regarding the video.

    Returns:
        str: The answer to the user's question or a message indicating that the transcript 
             has not been fetched.
    """
    global fetched_transcript, processed_transcript

    # Check if the transcript needs to be fetched
    if not processed_transcript:
        if video_url:
            # Fetch and preprocess transcript
            fetched_transcript = get_transcript(video_url)
            processed_transcript = process(fetched_transcript)
        else:
            return "Please provide a valid YouTube URL."

    if processed_transcript and user_question:
        # Step 1: Chunk the transcript (only for Q&A)
        chunks = chunk_transcript(processed_transcript)

        # Step 2: Set up llm and embeddings (only for Q&A)
        llm = model

        # Step 3: Create FAISS index for transcript chunks (only needed for Q&A)
        embedding_model = embeddings
        faiss_index = create_faiss_index(chunks, embedding_model)

        # Step 4: Set up the Q&A prompt and chain
        qa_prompt = create_qa_prompt_template()
        qa_chain = create_qa_chain(llm, qa_prompt)

        # Step 5: Generate the answer using FAISS index
        answer = generate_answer(user_question, faiss_index, qa_chain)
        return answer
    else:
        return "Please provide a valid question and ensure the transcript has been fetched."

# Setting up the Gradio interface
with gr.Blocks() as interface:
    gr.Markdown(
        "<h2 style='text-align: center;'>YouTube Video Summarizer and Q&A</h2>"
    )
    # Input field for YouTube URL
    video_url = gr.Textbox(label="YouTube Video URL", placeholder="Enter the YouTube Video URL")
    
    # Outputs for summary and answer
    summary_output = gr.Textbox(label="Video Summary", lines=5)
    question_input = gr.Textbox(label="Ask a Question About the Video", placeholder="Ask your question")
    answer_output = gr.Textbox(label="Answer to Your Question", lines=5)

    # Buttons for selecting functionalities after fetching transcript
    summarize_btn = gr.Button("Summarize Video")
    question_btn = gr.Button("Ask a Question")

    # Display status message for transcript fetch
    transcript_status = gr.Textbox(label="Transcript Status", interactive=False)

    # Set up button actions
    summarize_btn.click(summarize_video, inputs=video_url, outputs=summary_output)
    question_btn.click(answer_question, inputs=[video_url, question_input], outputs=answer_output)

# Launch the app with specified server name and port
interface.launch(server_name="0.0.0.0", server_port=7860,share=True)
