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

from predicto import get_digits
from predictc import get_prediction_data

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

# function to pick a random file from a folder
# if there is only one, it will return it
def pick_random_file(folder_path):
    try:
        # Get a list of all files in the folder
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
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
In the center of an oval are a digit then a tree with leaves to the top then a digit.\
Determine the 2 digits on either side of the tree.\
Be sure not to mistake the trunk of the tree for a digit.\
Return the 2 digits as a single string and in json format \
using the key of 'pred', with no other text, not even the text 'json'.\
like this:  {"pred":55}
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
                # print(f"no image in {folder_path} - sleeping 5 seconds...")
                # time.sleep(5)
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

            thread1 = threading.Thread(target=get_prediction_data, args=(learner, img_path, output))
            thread2 = threading.Thread(target=get_digits, args=(prompt, img_path, output))

            logging.info(f"before threads started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            thread1.start()
            thread2.start()

            thread1.join()
            thread2.join()

            logging.info(f"after threads joined: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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