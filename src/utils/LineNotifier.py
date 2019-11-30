# -*- coding: utf-8 -*-
import requests
import json


class LineNotifier:
    def __init__(self):
        with open('utils.json') as f:
            self.token = json.load(f)["ln"]["token"]
        self.url = "https://notify-api.line.me/api/notify"
        self.headers = {"Authorization": "Bearer " + self.token}

    def send_message(self, message):
        payload = {"message": message}
        try:
            r = requests.post(self.url, headers=self.headers, params=payload)
        except ConnectionError:
            pass


if __name__ == "__main__":
    ln = LineNotifier()
    ln.send_message("test")
