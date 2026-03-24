from openreward.environments import Server

from atc import ATCEnvironment

if __name__ == "__main__":
    server = Server([ATCEnvironment])
    server.run()
