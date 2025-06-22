from classify import classify_remote_image
import time

def main():
    while (True):
        url = "https://80b2-2607-f140-400-68-3006-f365-d41f-e5a6.ngrok-free.app/screenshot_full"
        classify_remote_image(url)
        time.sleep(1)

main()