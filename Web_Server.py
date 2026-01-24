import asyncio
import json
import websockets
from threading import Thread

class LiveWebServer:
    def __init__(self):
        self.clients = set()

    async def handler(self, websocket):
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)

    async def broadcast(self, data):
        if self.clients:
            msg = json.dumps(data)
            await asyncio.gather(
                *[client.send(msg) for client in self.clients]
            )

    async def run(self):
        async with websockets.serve(self.handler, "0.0.0.0", 8765):
            await asyncio.Future()

def start_web_server(server):
    asyncio.run(server.run())

web_server = LiveWebServer()
Thread(target=start_web_server, args=(web_server,), daemon=True).start()
