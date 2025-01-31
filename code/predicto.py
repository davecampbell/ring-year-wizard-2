import openai
import base64
import json

def get_digits(prompt, image_path):
  with open(image_path, "rb") as image_file:
    image_data = image_file.read()
    
  response = openai.chat.completions.create(
  model="gpt-4o",
  messages=[
    {
      "role": "user",
      "content": [
          {"type": "text", "text": prompt},
          {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(image_data).decode()}"}} # Encode image_data to base64
          ]
      }
    ],
  max_tokens=300,
  temperature=0.0
  )
  # print(response.choices[0].message.content)
  json_string = response.choices[0].message.content
  digits = json.loads(json_string)
  return digits

