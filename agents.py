from crewai import Agent
from tools import tool
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


# ## Call the gemini model
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
#                              verbose=True,
#                              temperature=0.5,
#                              google_api_key=os.getenv("GOOGLE_API_KEY"))

llm = Ollama(
    model="llama3",
    temperature=0.01,  # Adjust temperature as needed
    verbose=True
)

# Test the llm to ensure it is working correctly
try:
    test_prompt = "Hello, how are you?"
    response = llm(test_prompt)
    print(f"Test response from Ollama LLM: {response}")
except Exception as e:
    print(f"Error testing Ollama LLM: {e}")

## Create a senior blog content researcher

blog_researcher=Agent(
    role='Blog Researcher from Youtube Videos',
    goal='get the relevant video transcription for the topic {topic} from the provided Yt channel',
    verbose=True,
    memory=True,
    backstory=(
       "Expert in understanding videos in AI Data Science , MAchine Learning And GEN AI and providing suggestion" 
    ),
    tools=[tool],
    llm=llm,
    # embedder=llm,
    allow_delegation=True
)

## creating a senior blog writer agent with YT tool

blog_writer=Agent(
    role='Blog Writer',
    goal='Narrate compelling tech stories about the video {topic} from YT video',
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft"
        "engaging narratives that captivate and educate, bringing new"
        "discoveries to light in an accessible manner."
    ),
    tools=[tool],
    llm=llm,
    # embedder=llm,
    allow_delegation=False


)