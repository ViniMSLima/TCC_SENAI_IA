import cv2
import numpy as np
import os
import time


def is_red_present(frame, threshold=10000):
    """
    Verifica se a quantidade de vermelho na imagem excede o limiar especificado.

    Args:
    frame: Frame da câmera em formato BGR.
    threshold: Limiar para a quantidade de pixels vermelhos.

    Returns:
    bool: True se a quantidade de vermelho exceder o limiar, False caso contrário.
    """
    # Converte o frame para o espaço de cores HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define intervalos para a cor vermelha no espaço HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Cria máscaras para as duas faixas de vermelho
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combina as máscaras
    mask = mask1 + mask2

    # Conta os pixels vermelhos
    red_pixels = np.sum(mask > 0)

    return red_pixels > threshold


def save_image(frame, save_dir, counter):
    """
    Salva o frame atual como uma imagem.

    Args:
    frame: Frame da câmera em formato BGR.
    save_dir: Diretório onde as imagens serão salvas.
    counter: Contador para nomear as imagens.

    Returns:
    str: Caminho da imagem salva.
    """
    image_path = os.path.join(save_dir, f'foto_{counter}.png')
    cv2.imwrite(image_path, frame)
    print(f'Foto salva: {image_path}')
    return image_path


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
    flash_color = (255, 255, 255)  # padrão branco
    red_detected = False  # estado de detecção de vermelho
    red_last_detected = False  # estado anterior de detecção de vermelho

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
                image_path = save_image(frame, save_dir, counter)
                counter += 1
                flash_frames = 5  # exibe o flash por 5 frames
                flash_color = (0, 255, 0)  # verde para indicar sucesso

            # Verifica se há uma quantidade significativa de vermelho na imagem
            red_detected = is_red_present(frame)

            # Verifica transição de estado (vermelho saindo e entrando novamente)
            if red_detected and not red_last_detected:
                # Espera 2 segundos exibindo a contagem na tela
                for i in range(2, 0, -1):
                    countdown_frame = frame.copy()
                    cv2.putText(countdown_frame, f'Tirando foto em {i}s', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow('CAM', countdown_frame)
                    cv2.waitKey(1000)  # espera 1 segundo

                # Tira a foto
                image_path = save_image(frame, save_dir, counter)
                counter += 1
                flash_frames = 5  # exibe o flash por 5 frames
                flash_color = (0, 255, 0)  # verde para indicar sucesso

            # Atualiza o estado anterior de detecção de vermelho
            red_last_detected = red_detected

            # Adiciona efeito de flash na tela
            if flash_frames > 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), flash_color, -1)
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
