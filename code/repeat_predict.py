import time
import argparse
import threading
import logging
import hashlib

import os
import subprocess
from dotenv import load_dotenv
import cv2
from datetime import datetime
import random

from predicto import get_digits

now = datetime.now()
formatted_datetime = now.strftime('%Y%m%d%H%M%S')

logging.basicConfig(
    filename="logs/app-" + formatted_datetime + ".log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Logging new session ==============================================================")

# set up the expected command line arguments
parser = argparse.ArgumentParser(description='Predict/Recognize the year on a class ring image.')
parser.add_argument(
    "-i", "--img_path",
    type=str,
    help='The path to the image to be predicted.'
)

parser.add_argument(
    "-m", "--mode",
    type=str,
    default="look",
    help='The mode of running this program. Options: "random" or "look".'
)

parser.add_argument(
    "-p", "--path",
    type=str,
    default="chatgpt",
    help='prediction execution path. Options: "chatgpt"'
)

parser.add_argument(
    "-d", "--debug",
    action='store_true',
    help='turn debug state on'
)

# parse the command line
args = parser.parse_args()
# get a count of the number of non-None arguments
num_args = sum(
        1 for arg in vars(args).values() if arg not in (None, False)
    )

def is_image(file_path):
    # List of valid image extensions
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    # Get the file extension
    file_extension = os.path.splitext(file_path)[1].lower()
    return file_extension in valid_extensions

# function to pick a random file from a folder
# if there is only one, it will return it
def pick_random_file(folder_path):

    try:
        # Get a list of all image files in the folder
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and is_image(f)]
        if not files:
            # print("No files found in the folder.")
            return None
        # Pick a random file
        random_file = random.choice(files)
        return random_file
    except FileNotFoundError:
        print("The specified folder does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def read_file_flag(file_path: str):
    """
    Reads a flag in a file.

    :param file_path: Path to the file containing the flag.
    :return: value of the flag.
    """
    with open(file_path, 'r') as file:
        for line in file:
            return line.strip()

def is_file_open(file_path):
    """
    Check if a file is open by any process using lsof.
    :param file_path: Path to the file
    :return: True if the file is open, False otherwise
    """
    try:
        result = subprocess.run(['lsof', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0  # If lsof returns 0, the file is open
    except Exception as e:
        # print(f"Error checking file with lsof: {e}")
        return False

# hashing routine used to tell if the image is the same as the previous image
# if so, don't burn the API tokens on a prediction
def get_image_hash(image_path, algorithm="md5"):
    """Compute hash of an image file using the specified hashing algorithm."""
    hasher = hashlib.new(algorithm)
    with open(image_path, "rb") as f:
        while chunk := f.read(4096):
            hasher.update(chunk)
    return hasher.hexdigest()


load_dotenv('.env')

# folder locations
default_folder_path = os.getenv("DEFAULT_FOLDER_PATH")
look_folder_path = os.getenv("LOOK_FOLDER_PATH")
output_path = os.getenv("OUTPUT_PATH")
flag_file_path = os.getenv("FLAG_FILE_PATH")


# initialize
last_image_hash = None

while True:

    while read_file_flag(flag_file_path) == 'GO':

        # response output
        output = {}

        # determine which image to predict on 
        if num_args == 0 or args.mode == 'random':
            folder_path = default_folder_path
            img = pick_random_file(folder_path)
            img_path = folder_path + img

        elif args.mode == 'look':
            folder_path = 'images/look/'
            img = pick_random_file(folder_path)
            while img is None:
                time.sleep(1)
                img = pick_random_file(folder_path)
            img_path = folder_path + img

        elif len(args.img_path) > 0:
            img_path = args.img_path

        else:
            print("confused state")
            break

        # get image hash
        current_image_hash = get_image_hash(img_path)
        
        if current_image_hash != last_image_hash:

            logging.info(f"predicting: {img_path}")
            output["img_path"] = img_path

            # pre-process image
            img = cv2.imread(img_path)
            # make grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            prompts = [None, None]

            prompts[0] = """
Analyze this image of a ring. Use the palmetto tree at the center as the reference for orientation. The tree is upright if the branches are above the trunk.\
First, determine the orientation of the image using the tree. Return "orient": 1 if the image is upright, "orient": 0 if flipped.\
Then, interpret the two single digits on either side of the tree trunk as they would appear in an upright image. If the image is flipped, rotate each digit 180° before interpreting.\
Be sure to not mistake the tree trunk for a digit.\
There is a single digit character then a tree trunk then a single digit character.\
Your result will be a string with 2 characters.\
Return your result as JSON using:\
"orient" for orientation, and "pred" for the final digit string.\
Example format: {"orient": 1, "pred": "57"}\
Return only the JSON formatted as above, no other text like 'json'.\
Never return any other text, especially 'json'.
"""

            prompts[1] =  """
Analyze this image of a ring.\
Interpret the two digits beside the tree, one digit character on each side of the tree trunk.\
Be sure to not mistake the tree trunk for a digit.\
There is a single digit character then a tree trunk then a single digit character.\
Your result will be a string with 2 characters.\
Return your result as JSON using "pred" for the digit string.\
Example format: {"pred": "57"}\
Return only the JSON, no other text like 'json'.\
Never return any other text, especially 'json'.
"""


            if args.path == 'chatgpt':

                # pipeline to first ask ChatGPT if the image is upright or not, then predict on that image if upright,
                # or a flipped image if not upright

                before = datetime.now()

                digits = get_digits(prompts, img)

                after = datetime.now()
                logging.info(f"chatgpt time: {after - before}")
                print(f"chatgpt time: {after - before}")

            # to a file
            # with open(output_path, "w") as file:
            #    json.dump(output, file, indent=4)  # `indent` makes the JSON more readable

            # to the console
            print(f"\n{digits}\n")

            # extra debug output (if debug is on)
            if args.debug:
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                print("CAN PRINT OUT MORE STUFF HERE")


            last_image_hash = current_image_hash

        else:
            logging.info(f"same image - sleeping 5 seconds...")
            time.sleep(5)