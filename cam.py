import cv2
import numpy as np
import os
import time

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
    image_path = os.path.join(save_dir, f'foto_{counter}.png')
    cv2.imwrite(image_path, frame)
    print(f'Foto salva: {image_path}')
    return image_path

def execute_another_script(script_path):
    with open(script_path, 'r') as script_file:
        script_content = script_file.read()
        exec(script_content, globals())

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
                    cv2.waitKey(1000)

                ret, frame = cap.read()
                if ret:
                    image_path = save_image(frame, save_dir, counter)
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
