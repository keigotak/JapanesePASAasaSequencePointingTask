import json
import requests

with open('../utils/utils.json') as f:
    ret = json.load(f)["gs"]

ENDPOINT = {"train": ret["train"],
            "test": ret["test"]}


def write_spreadsheet(*args, type="train"):
    endpoint = ENDPOINT[type]
    text = [str(item) for item in args[0]]
    requests.post(endpoint, json.dumps(text))


if __name__ == "__main__":
    write_spreadsheet(["test"], type="train")
    write_spreadsheet(["test"], type="test")
