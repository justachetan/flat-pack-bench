import os
from google import genai
from google.genai import types
import dotenv
dotenv.load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), vertexai=False)

# response = client.models.generate_content(
#     model="gemini-2.5-pro",
#     contents="How does AI work?",
#     config=types.GenerateContentConfig(
#         system_instruction="Explain AI in simple terms.",
#         thinking_config=types.ThinkingConfig(thinking_budget=-1) # Disables thinking
#     ),
# )
# print(response.text)
# total_size = 0
# print("My files:")
# for f in client.files.list():
#     print("  ", f.name, f.size_bytes / (10**9), "GB")
#     total_size += f.size_bytes
# print("Total size:", total_size / (10**9), "GB")

