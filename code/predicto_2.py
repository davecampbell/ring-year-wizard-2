import openai
import base64
import json

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
def get_digits(prompt, image_paths, output):
  image_1 = encode_image(image_paths[0])
  image_2 = encode_image(image_paths[1])
    
  response = openai.chat.completions.create(
  model="gpt-4o",
  messages=[
    {
      "role": "user",
      "content": [
         {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_1}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_2}"}}
          ]
      }
    ],
  max_tokens=500,
  temperature=0.0
  )
  # print(response.choices[0].message.content)
  json_string = response.choices[0].message.content
  resp = json.loads(json_string)
  
  output["open_ai"] = {}
  output["open_ai"]["pred"] = resp["lefty"] + resp["righty"]
  output["open_ai"]["lefty"] = resp["lefty"]
  output["open_ai"]["righty"] = resp["righty"]
  output["open_ai"]["up_pic"] = resp["up_pic"]

  return output["open_ai"]

