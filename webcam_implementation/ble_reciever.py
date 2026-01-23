import asyncio
import json
from bleak import BleakScanner
from ws import web_server

TARGET_MAC = "D8:3A:DD:02:4F:26"  # set to None to accept all devices

def advertisement_callback(device, adv):
    if TARGET_MAC and device.address != TARGET_MAC:
        return

    for company_id, mfg_data in adv.manufacturer_data.items():
        payload = bytes(mfg_data)

        loop = asyncio.get_running_loop()

        x = int.from_bytes(payload[0:2], byteorder="big", signed=False)
        y = int.from_bytes(payload[2:4], byteorder="big", signed=False)
        possession = payload[4]
        score_black = payload[5]
        score_white = payload[6]
        goal_event = payload[7]
        speed = int.from_bytes(payload[8:10], byteorder="big", signed=False)
        frame = payload[10]
       
        loop.create_task(
            web_server.broadcast({
                "x": x,
                "y": y,
                "speed": speed,
                "possession": possession,
                "score_black": score_black,
                "score_white": score_white,
                "goal_event": goal_event,
                "frame": frame
            })
        )

async def main():
    print("Starting BLE scan...")
    scanner = BleakScanner(advertisement_callback)
    await scanner.start()

    try:
        while True:
            await asyncio.sleep(1)
    finally:
        await scanner.stop()


if __name__ == "__main__":
    asyncio.run(main())