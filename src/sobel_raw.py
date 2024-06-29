import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from time import perf_counter_ns

def apply_sobel_filter(img):
    #Matrizes de gradiente utilizadas para encontrar as "edges" na imagem,
    #conforme a formula do filtro de Sobel
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    #Inicializa as imagens com os gradientes x e y
    grad_img_x = np.zeros_like(img)
    grad_img_y = np.zeros_like(img)

    #Aplica o filtro
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            grad_img_x[i, j] = np.sum(Gx * img[i - 1:i + 2, j - 1:j + 2])
            grad_img_y[i, j] = np.sum(Gy * img[i - 1:i + 2, j - 1:j + 2])

    #Normaliza a magnitude: pitagoras -> √(grad_img_x^2 + grad_img_y^2)
    magnitude = np.hypot(grad_img_x, grad_img_y)
    #Coloca essa magnitude num range de 0 a 255 (8 bits -> imagem em preto e branco)
    magnitude = np.clip(magnitude, 0, 255)
    #Retorna a magnitude no formato de unsigned integer de 8 bits
    return magnitude.astype(np.uint8)

def is_rgba_img(img):
    # A funcao imread retorna no shape[2] o valor de 4 caso a imagem seja RGBA
    return img.ndim == 3 and img.shape[2] == 4
def is_rgb_img(img):
    # A funcao imread retorna no shape[2] o valor de 3 caso a imagem seja RGB
    return img.ndim == 3 and img.shape[2] == 3

def convert_to_rgb(img):
    #Codigo de 'alpha blending', para converter uma imagem RGBA para RGB
    img_rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype='float32')
    r, g, b, a = img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]
    R, G, B = (1, 1, 1)
    img_rgb[:, :, 0] = r * a + (1.0 - a) * R
    img_rgb[:, :, 1] = g * a + (1.0 - a) * G
    img_rgb[:, :, 2] = b * a + (1.0 - a) * B
    return np.asarray(img_rgb, dtype='float32')


def convert_to_grayscale(img):
    #Formula para converter imagem colorida em grayscale:
    # R * 299/1000 | G * 587/1000 | B * 114/1000
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

def main():
    #Pegar a imagem
    img_path = '../img/Paisagem.jpg'
    input_img = mpimg.imread(img_path)
    #Inicializa a variavel da imagem com o filtro
    sobel_img = None

    #Se a imagem for RGBA, temos que converter para RGB
    if is_rgba_img(input_img):
        input_img = convert_to_rgb(input_img)

    #Se for uma imagem RGB, temos que tranforma-la em preto e branco (grayscale)
    if is_rgb_img(input_img):
        input_img = convert_to_grayscale(input_img)

    #Converter para float32
    input_img = input_img.astype(np.float32)

    runs = 1
    timer = np.empty(runs, dtype=np.float32)
    for i in range(timer.size):
        tic = perf_counter_ns()
        sobel_img = apply_sobel_filter(input_img)
        toc = perf_counter_ns()
        timer[i] = toc - tic
    timer *= 1e-6
    print(f"Elapsed time: {timer.mean():.3f} +- {timer.std():.3f} ms")

    #Mostra o resultado
    plt.rcParams["figure.figsize"] = (8, 8)
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.imshow(sobel_img, cmap='gray')
    plt.title('Sobel Filter Result')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()