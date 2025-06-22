from classify import classify_remote_image
import time

def main():
    while (True):
        url = "https://6285-2607-f140-400-68-c90d-840b-94de-89bc.ngrok-free.app/screenshot_full"
        classify_remote_image(url)
        time.sleep(1)

main()