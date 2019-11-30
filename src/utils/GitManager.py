# -*- coding: utf-8 -*-
import subprocess


class GitManager:
    def __init__(self):
        self.sha = self.get_sha()

    def get_sha(self):
        cmd = "git show -s --format=%H".split()
        return subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode("utf8").strip()


if __name__ == "__main__":
    gm = GitManager()
    print(gm.get_sha())
