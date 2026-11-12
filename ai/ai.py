from google import genai
from dotenv import load_dotenv
import os, logging, json

# hide warning messages from the Gemini SDK which uses Python's logging module
load_dotenv()
logging.getLogger().setLevel(logging.ERROR)

def build_prompt(behaviour_data):
    prompt = f"""
    You are an AI assistant specialising in animal behaviour and body language interpretation.
    Your task is to analyse the behavioural observations provided by the Behaviour Logic system and interpret what the animal might be doing, feeling, or communicating through its body language.
    Identify the animal's likely behaviour, interpret what its body language might mean, and provide a short, natural-language explanation that helps the user understand the animal.

    Base your interpretation strictly on the provided observations. Consider data quality, detection reliability, and changes in behaviour over time. Do not invent observations or treat uncertain movements as definitive evidence of emotions.
    If the information is insufficient, communicate the uncertainty instead of making assumptions.
    
    Analyze this behaviour data: {behaviour_data}
    
    Return your interpretation in the following JSON format:

    {{
    "behaviour": "What the animal appears to be doing",
    "body_language_meaning": "What its body language might be expressing",
    "explanation": "A short, user-friendly interpretation (maximum 25 words)",
    "supporting_observations": [
        "Key observations supporting the interpretation"
    ],
    "confidence": "High | Medium | Low",
    "data_quality": "Good | Limited | Poor"
    }}

    Return only a valid JSON object without additional text.
    """
    return prompt


def get_ai_output(prompt):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL"),
        contents=prompt
    )
    return response.text

def get_ai_output_retry(prompt):
    print(f"Retrying connection to Google {os.getenv("GEMINI_MODEL")} API once...")
    try:
        return get_ai_output(prompt)
    except Exception as error:
        print("The API is still unavailable.")
        return False

def validate_ai_output(ai_output):
    try:
        result = json.loads(ai_output) # store ai_output in a JSON object
        required_fields = [
            "behaviour",
            "body_language_meaning",
            "explanation",
            "supporting_observations",
            "confidence",
            "data_quality"
        ]

        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing field: {field}")
        return result

    except (json.JSONDecodeError, ValueError) as error:
        logging.error("Failed to parse AI response: %s", error)
        return None
    

def find_json_file():
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file.endswith("summary.json"):
                return os.path.join(root, file)
    return None    


def read_json(data_file):
    try:
        with open(data_file, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print("JSON file not found.")
    except json.JSONDecodeError:
        print("Invalid JSON format.")
        
    
def analyze_data(behaviour_data):
    if not isinstance(behaviour_data, dict):
        raise TypeError("behaviour_data must be a dictionary")
    
    prompt = build_prompt(behaviour_data) # get the prompt
    
    try:
        print("Analyzing cat behavior data...")        
        ai_output = get_ai_output(prompt) # get the AI output from Gemini
        print("Cat behavior analysis complete. Here is the output:\n")
    except Exception:
        print("Gemini failed to interpret the data!")
        ai_output = get_ai_output_retry(prompt)
    
    # when Gemini API is unavailable to use after retry attempt, pass None to IO side to display on Tinycircuit
    if ai_output is None:
        return None
        
    result = validate_ai_output(ai_output) # validate AI output when it is valid
    
    if result:
        # print(result)
        return result