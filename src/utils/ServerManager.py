from socket import gethostname


class ServerManager:
    def __init__(self, device_name="cpu"):
        self.server_name = '[%s]' % gethostname()
        self.device_name = device_name

    def get_server_name(self):
        return self.server_name

    def get_device_name(self):
        return self.device_name


if __name__ == "__main__":
    sm = ServerManager()
    print(sm.get_server_name())
    print(sm.get_device_name())