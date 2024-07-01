import cv2
import numpy as np
import os
import time
import requests
import json

def is_red_present(frame, threshold=10000):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    red_pixels = np.sum(mask > 0)
    return red_pixels > threshold

def save_image(frame, save_dir, counter):
    image_path = os.path.join(save_dir, '0.png')
    cv2.imwrite(image_path, frame)
    print(f'Foto salva: {image_path}')
    return image_path

def resize_and_process_image(image_path, output_dir, index, size=(128, 128)):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img is None:
        print(f"Erro ao ler a imagem: {image_path}")
        return

    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(img_hsv, lower_red, upper_red)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    img_red_filtered = cv2.bitwise_and(img, img, mask=mask)
    img_gray = cv2.cvtColor(img_red_filtered, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_resized = cv2.resize(img_binary, size)

    output_path = os.path.join(output_dir, "0.png")
    cv2.imwrite(output_path, img_resized, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"Imagem salva: {output_path}")
    return output_path

def execute_another_script(script_path):
    with open(script_path, 'r') as script_file:
        script_content = script_file.read()
        exec(script_content, globals())

def send_image_to_server(image_path):
    url = 'http://127.0.0.1:5000/json/'  # Update with your server's address
    data = {
        'images': ["captured_images/prediction_test/0.png"]
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        print(f"Server response: {response.json()}")
    else:
        print(f"Failed to get response from server. Status code: {response.status_code}")

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera.")
        return

    save_dir = 'captured_images/prediction_test'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    counter = 0
    flash_frames = 0
    flash_color = (255, 255, 255)
    red_detected = False
    red_last_detected = False

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Erro: Não foi possível capturar o frame.")
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                image_path = save_image(frame, save_dir, counter)
                processed_image_path = resize_and_process_image(image_path, save_dir, counter)
                send_image_to_server(processed_image_path)
                counter += 1
                flash_frames = 5
                flash_color = (0, 255, 0)
                execute_another_script('Training_AI/transform.py')

            red_detected = is_red_present(frame)

            if red_detected and not red_last_detected:
                for i in range(2, 0, -1):
                    countdown_frame = frame.copy()
                    cv2.putText(countdown_frame, f'Tirando foto em {i}s', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow('CAM', countdown_frame)
                    cv2.waitKey(1300)

                ret, frame = cap.read()
                if ret:
                    image_path = save_image(frame, save_dir, counter)
                    processed_image_path = resize_and_process_image(image_path, save_dir, counter)
                    send_image_to_server(processed_image_path)
                    counter += 1
                    flash_frames = 5
                    flash_color = (0, 255, 0)
                    execute_another_script('Training_AI/transform.py')

            red_last_detected = red_detected

            if flash_frames > 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), flash_color, -1)
                alpha = 0.5
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                flash_frames -= 1

            cv2.imshow('CAM', frame)

            if key == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrompido pelo usuário")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
