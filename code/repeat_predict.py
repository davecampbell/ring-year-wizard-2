import warnings

# this suppresses from warnings about torchvision
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from fastai.vision.all import *
    import torch

import time
import argparse
import threading
import logging
import hashlib

import os
import subprocess
from dotenv import load_dotenv
import cv2

from predicto import get_digits
from predicto_2 import get_digits as get_digits_2
from predictc import get_prediction_data
from predictc_2 import get_prediction_data as get_prediction_data_2
from image_utils import flip_vertically, ellipsify

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
    help='The mode of running this program. Options: "random" or "look".'
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

def year_from_name(fname):
    pattern = r'(?<=\d{2})\d{2}(?=_)'
    digits = re.search(pattern, str(fname)).group()
    return digits

# this is a custom function that was used in the datablock of the learner
# need to define it here so it can be monkey-patched into the model after
# the exported version
def custom_labeller(fname):
    # Assuming filenames are like "classX_imageY.jpg"
    # return fname.name.split('_')[0]  # Extract class name based on your logic
    pattern = r'(?<=\d{2})\d{2}(?=_)'
    digits = re.search(pattern, str(fname)).group()
    klasses = list(digits)
    klasses.append(digits)
    return klasses

# the monkey-patch
globals()['custom_labeller'] = custom_labeller

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

custom_model_path = os.getenv("CUSTOM_MODEL_PATH")

# Load the custom model
learner = load_learner(custom_model_path)

prompt = """
These 2 monochrome images are the same image, flipped vertically. \
Be sure to look at both images before analyzing. \
The first task is to determine which image is upright. \
There is a large C and a tree in the image to help decide \
which image is upright. \
In the upright image, the C will be open to the right and the tree will have \
a trunk pointing down and leaves and branches pointing up. \
In the flipped image, the C will be open to the left and the tree will have \
a trunk pointing up and leaves and branches pointing down. \
First, decide which image is upright, then use that image to do the next step. \
If the first image is the upright image, set 'up_pic' = 0.\
If the second image is the upright image, set 'up_pic' = 1.\
If neither image shows a C and a tree, return the number '39' in the normal \
format and set 'lefty' = 'Y' and 'righty' = 'Z'. \
Within the middle of the 'C' in the upright image are three symbols - \
left-to-right - a digit (0-9) we are calling 'lefty' then a tree symbol, then \
a digit (0-9) we are calling 'righty'. \
The most important task is to determine 'lefty; and 'righty' on either side of \
the tree symbol. \
the digit symbols may be blotchy, they are engraved.  try hard to determine the \
digit symbols.\
store the digit to the left of the tree trunk as 'lefty' and the digit to \
the right of the tree trunk as 'righty'. \
if there is confusion about 'lefty' and 'righty', set it to 'X' but you must \
return an answer. \
In all cases, you must return an answer in the proper format, no excpetions. \
Return the results ('lefty', right' and 'up_pic') like this: \
{"lefty": '5', "righty":'7', "up_pic":1} \
and no other text, not even the text 'json'. \
You must return an answer formatted like this EVERY TIME. /
"""

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
            e_img = ellipsify(img)
            cv2.imwrite("images/tmp/e_img.jpg", e_img)
            flip_img = flip_vertically(e_img)
            cv2.imwrite("images/tmp/flip_img.jpg", flip_img)

            thread1 = threading.Thread(target=get_prediction_data, args=(learner, "images/tmp/e_img.jpg", output))
            thread2 = threading.Thread(target=get_digits_2, args=(prompt, ("images/tmp/e_img.jpg", "images/tmp/flip_img.jpg"), output))

            logging.info(f"before threads started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            thread1.start()
            thread2.start()

            thread1.join()
            thread2.join()

            logging.info(f"after threads joined: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # pipeline to use openai to determine the UPRIGHT image, then pass that image to custom model
            openai_output = get_digits_2(prompt, ("images/tmp/e_img.jpg", "images/tmp/flip_img.jpg"), output)
            if openai_output["up_pic"] == 0:
                img_path = "images/tmp/e_img.jpg"
            elif openai_output["up_pic"] == 1:
                img_path = "images/tmp/flip_img.jpg"

            upright_results = get_prediction_data_2(learner, img_path, output)

            output["upright_custom"] = upright_results

            logging.info(f"output: {output}")

            # print(output if debug is on)
            if args.debug:
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                print("CAN PRINT OUT MORE STUFF HERE")

            # write the output to a file
            with open(output_path, "w") as file:
                json.dump(output, file, indent=4)  # `indent` makes the JSON more readable

            last_image_hash = current_image_hash

        else:
            logging.info(f"same image - sleeping 5 seconds...")
            time.sleep(5)