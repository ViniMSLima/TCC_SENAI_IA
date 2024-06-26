import os
import cv2 as cv
import numpy as np

def resize_and_process_image(image_path, output_dir, index, size=(128, 128)):
    img = cv.imread(image_path, cv.IMREAD_COLOR)

    if img is None:
        print(f"Erro ao ler a imagem: {image_path}")
        return

    # Aplica um filtro Gaussiano para suavizar a imagem
    img_blur = cv.GaussianBlur(img, (5, 5), 0)

    # Converte a imagem suavizada para HSV
    img_hsv = cv.cvtColor(img_blur, cv.COLOR_BGR2HSV)

    # Define a faixa de cor vermelha na escala HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Cria a máscara para a cor vermelha
    mask = cv.inRange(img_hsv, lower_red, upper_red)

    # Aplica operações morfológicas para remover ruídos
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # Aplica a máscara à imagem original para obter apenas as partes vermelhas
    img_red_filtered = cv.bitwise_and(img, img, mask=mask)

    # Redimensiona a imagem resultante
    img_resized = cv.resize(img_red_filtered, size)

    # Normaliza os valores dos pixels para o intervalo [0, 1]
    img_resized = img_resized / 255.0

    # Salva a imagem resultante em formato PNG
    cv.imwrite(f"{index}.png", img_resized * 255, [cv.IMWRITE_PNG_COMPRESSION, 9])

def process_images_from_directory(directory, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Inicializa o contador de imagens
    image_index = 0

    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".png"):
                img_path = os.path.join(subdir, file)
                filename = os.path.splitext(file)[0]
                resize_and_process_image(img_path, output_dir, image_index)
                image_index += 1  # Incrementa o contador de imagem após processamento

directory = 'captured_images'
output_directory = 'AI/datasets/OUT/images'
input_dir = os.path.join(directory, "test")
process_images_from_directory(input_dir, output_directory)
