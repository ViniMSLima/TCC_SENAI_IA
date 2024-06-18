import cv2
import os
import time

def main():
    # Captura de vídeo da câmera padrão (0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera.")
        return

    # Cria uma pasta para salvar as fotos
    save_dir = 'captured_images'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    counter = 0
    flash_frames = 0  # contador de frames para o efeito de flash

    try:
        while True:
            # Captura frame a frame
            ret, frame = cap.read()

            if not ret:
                print("Erro: Não foi possível capturar o frame.")
                break

            # Verifica se a tecla espaço foi pressionada
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                image_path = os.path.join(save_dir, f'foto_{counter}.png')
                cv2.imwrite(image_path, frame)
                print(f'Foto salva: {image_path}')
                counter += 1
                flash_frames = 5  # exibe o flash por 5 frames

            # Adiciona efeito de flash na tela
            if flash_frames > 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255), -1)
                alpha = 0.5  # transparência do flash
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                flash_frames -= 1

            # Exibe o frame resultante
            cv2.imshow('CAM', frame)

            # Pressione 'q' no teclado para sair do loop
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrompido pelo usuário")

    # Libera a captura de vídeo
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
