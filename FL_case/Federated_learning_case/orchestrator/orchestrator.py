from websocket_server import WebsocketServerWorker

def main():
    server_worker = WebsocketServerWorker(host = "192.168.1.149",port=8765)
    server_worker.start()

main()