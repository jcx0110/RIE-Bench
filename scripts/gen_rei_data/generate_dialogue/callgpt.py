import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

import os
from openai import OpenAI

# API key from env OPENAI_API_KEY
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

from IPython.display import Image, display
import base64

# Path to the image
IMAGE_PATH = "data/sample_image.png"

# Preview the image
def preview_image(image_path):
    display(Image(image_path))

preview_image(IMAGE_PATH)

# Function to encode image to Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image(IMAGE_PATH)

# Base64 Image Processing
response_base64 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are an intelligent assistant that provides image summaries."},
        {"role": "user", "content": [
            {"type": "text", "text": "Please summarize the image content."},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }}
        ]}
    ],
    temperature=0.0,
)

print("Base64 Response:")
print(response_base64.choices[0].message.content)

# URL Image Processing
image_url = "https://example.com/sample_image.png"

response_url = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are an intelligent assistant that provides image summaries."},
        {"role": "user", "content": [
            {"type": "text", "text": "Please summarize the image content."},
            {"type": "image_url", "image_url": {
                "url": image_url
            }}
        ]}
    ],
    temperature=0.0,
)

print("URL Response:")
print(response_url.choices[0].message.content)
