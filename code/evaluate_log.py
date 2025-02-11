# this script evaluates the log file output from the predict script
# intended to have log output piped into it
# the line it wants to see is one with "output: " in it


import sys
import ast
import json
import re

def year_from_name(fname):
    pattern = r'(?<=\d{2})\d{2}(?=_)'
    digits = re.search(pattern, str(fname)).group()
    return digits

def main():
    line_count = 0

    c_right = 0
    c_wrong = []
    o_right = 0
    o_wrong = []

    # get the file match logic off the command-line if they are there
    # defaults to include al files
    args = sys.argv[1:]
    match_pattern = args[0] if len(args) > 0 else r'.*'
    MATCH_LOGIC = args[1] if len(args) > 1 else "include"


    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            # pull out just the part of line that is to the right of "output: " using regex
            line = line.split("output: ")[1]
            # replace all single-quotes with double-quotes
            line = line.replace("'", '"')
            # convert to json
            out = json.loads(line)

            # print(f"out: {out}")
            # print(f"out['img_path']: {out['img_path']}")
            # print(f"match_pattern: {match_pattern}")
            # print(f"match_pattern: {match_pattern}, search: {re.search(match_pattern, out['img_path'])}")

            # check filename match logic
            match = re.search(match_pattern, out["img_path"])
            # print(f"match: {bool(match)}, MATCH_LOGIC: {MATCH_LOGIC=='include'}")
            if bool(match) == (MATCH_LOGIC=="include"):

                line_count += 1

                year = year_from_name(out["img_path"])

                print(f"{out["img_path"]}, year: {year}, custom_model: {out['custom_model']}, open_ai:{out['open_ai']}")
                
                if out["custom_model"] == year:
                    c_right += 1
                else:
                    c_wrong.append(out["img_path"])

                if out["open_ai"] == year:
                    o_right += 1
                else:
                    o_wrong.append(out["img_path"])


        except Exception as e:
            print(f"Error processing line {line_count}: {e}")
            continue

    if line_count > 0:
        print(f"match_pattern: {match_pattern}, MATCH_LOGIC: {MATCH_LOGIC}")
        print(f"Custom model right: {int(c_right)/line_count}, OpenAI right: {int(o_right)/line_count}")
        print(f"Custom model wrong: {'\n'.join(c_wrong)}")
        print(f"OpenAI wrong: {'\n'.join(o_wrong)}")
    else:
        print("No lines processed")

if __name__ == "__main__":
    main()