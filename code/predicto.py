import openai
import base64
import json
import cv2

from image_utils import flip_vertically

def convert_img__to_png(img):
  # Encode the grayscale image as PNG to memory
  success, buffer = cv2.imencode('.png', img)
  if not success:
    raise ValueError("Image encoding failed")

  # Convert the buffer to base64
  return base64.b64encode(buffer).decode()

def get_digits(prompts, img):
    
  response = openai.chat.completions.create(
  model="gpt-4o",
  messages=[
    {
      "role": "user",
      "content": [
          {"type": "text", "text": prompts[0]},
          {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{convert_img__to_png(img)}"}} # Encode image_data to base64
          ]
      }
    ],
  max_tokens=300,
  temperature=0.0
  )
  print(response.choices[0].message.content)
  json_string = response.choices[0].message.content
  up_pic = json.loads(json_string)["orient"]
  digits = json.loads(json_string)["pred"]


  if up_pic == 0:
    img2 = flip_vertically(img)

    response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
      {
        "role": "user",
        "content": [
            {"type": "text", "text": prompts[1]},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{convert_img__to_png(img2)}"}} # Encode image_data to base64
            ]
        }
      ],
    max_tokens=300,
    temperature=0.0
    )  
    print(response.choices[0].message.content)
    json_string = response.choices[0].message.content
  
    digits = json.loads(json_string)["pred"]
  return digits

