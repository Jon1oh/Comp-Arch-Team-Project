from google import genai
from dotenv import load_dotenv
import os, logging

# hide warning messages from the Gemini SDK which uses Python's logging module
load_dotenv()
logging.getLogger().setLevel(logging.ERROR)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
interaction = client.interactions.create(
    model=os.get_env("GEMINI_MODEL"),
    input="Explain how AI works in a few words"
)

# print(interaction.output_text)

def build_prompt():
    return "test prompt"

def interpret_data():
    prompt = build_prompt()
    response = client.models.generate_content(
        model=os.get_env("GEMINI_MODEL"),
        contents=prompt
    )

    return response.text