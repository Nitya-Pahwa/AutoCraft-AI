import os

from dotenv import load_dotenv

load_dotenv()


def get_llm():
    """Optional LLM helper. The v1 pipeline works without an API key."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    from langchain_groq import ChatGroq

    return ChatGroq(
        groq_api_key=api_key,
        model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0.1,
    )

