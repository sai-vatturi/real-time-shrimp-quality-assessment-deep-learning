# client.py

import requests
import json
from pprint import pprint

def send_request(file_list=['./image1.jpg'],
                img_size=640,
                download_image=False):
    '''
    Sends images to the server for detection and prints the JSON response
    '''
    url = "http://localhost:8000/detect"

    # Prepare files and data
    files = []
    for file_path in file_list:
        try:
            f = open(file_path, "rb")
            files.append(('file_list', (file_path, f, 'image/jpeg')))
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return

    data = {
        'img_size': img_size,
        'download_image': download_image
    }

    try:
        response = requests.post(url, data=data, files=files)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return
    finally:
        # Ensure all files are closed
        for _, file in files:
            file[1].close()

    try:
        json_data = response.json()
    except json.JSONDecodeError:
        print("Failed to decode JSON response.")
        return

    if isinstance(json_data, dict) and 'error' in json_data:
        print(f"Error from server: {json_data['error']}")
        return

    pprint(json_data)

if __name__ == '__main__':
    send_request(file_list=['./image1.jpg'], download_image=True)
