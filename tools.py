from crewai_tools import YoutubeChannelSearchTool
from langchain_community.llms import Ollama

# # Initialize the tool with a specific Youtube channel handle to target your search
# tool = YoutubeChannelSearchTool(youtube_channel_handle='@krishnaik06')


tool = YoutubeChannelSearchTool(
    config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama3",
                temperature=0.5,
                top_p=1,
                stream=True,
            ),
        ),
        embedder=dict(
            provider="ollama", # or openai, ollama, ...
            config=dict(
                model="llama3",
                # task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)

# tool = YoutubeChannelSearchTool(
#     config=dict(
#         llm=dict(
#             provider="ollama", # or google, openai, anthropic, llama2, ...
#             config=dict(
#                 model="llama3",
#                 temperature=0.5,
#                 top_p=1,
#                 stream=True,
#             )
#         )
#     )
# )
