import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from numba import cuda, void, float32, uint8, int32
from time import perf_counter_ns

@cuda.jit(void(float32[:,:], uint8[:, :], int32[:, :], int32[:, :]))
def apply_sobel_filter_cuda(in_image, out_image, kernel_x, kernel_y):
    i, j = cuda.grid(2)
    height, width = in_image.shape
    kernel_height, kernel_width = kernel_x.shape

    # Calcular as margens pra evitar erro de out of bounds
    margin_y, margin_x = kernel_height // 2, kernel_width // 2

    if (margin_y <= i < height - margin_y) and (margin_x <= j < width - margin_x):
        Gx, Gy = 0, 0

        # Aplicar o algoritmo de Sobel
        for u in range(-margin_y, margin_y + 1):
            for v in range(-margin_x, margin_x + 1):
                img_val = in_image[i + u, j + v]
                Gx += kernel_x[u + margin_y, v + margin_x] * img_val
                Gy += kernel_y[u + margin_y, v + margin_x] * img_val

        # Calcular a magnitude do gradiente
        out_image[i, j] = min(255, max(0, math.sqrt(Gx ** 2 + Gy ** 2)))

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

    #Definir o tamanho do grid e bloco - CUDA:
    #Cada bloco sera composto por 16x16 threads
    threads = (16, 16)
    #Ja a quantidade de blocos por grid, sera feita com base no tamanho da imagem
    blocks_x = int(np.ceil(input_img.shape[0] / threads[0]))
    blocks_y = int(np.ceil(input_img.shape[1] / threads[1]))
    blocks = (blocks_x, blocks_y)

    # Constantes que enviaremos para o 'device' (GPU)
    SOBEL_X = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    SOBEL_Y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    def execute_sobel():
        nonlocal sobel_img
        #Copiar a imagem de entrada para o device
        input_img_device = cuda.to_device(input_img)
        #Alocar memoria para o output dentro do device
        output_img_device = cuda.device_array(input_img.shape, np.uint8)
        #Copiar as constantes para o device
        d_sobel_x = cuda.to_device(SOBEL_X)
        d_sobel_y = cuda.to_device(SOBEL_Y)

        #Chamada da funcao CUDA
        apply_sobel_filter_cuda[blocks, threads](input_img_device, output_img_device, d_sobel_x, d_sobel_y)
        #Copiar os resultados de volta para o host (CPU)
        cuda.synchronize()
        sobel_img = output_img_device.copy_to_host()
        #Limpar as variaveis do device
        del output_img_device
        del input_img_device
        del d_sobel_x
        del d_sobel_y

    runs = 1
    timer = np.empty(runs, dtype=np.float32)
    for i in range(timer.size):
        cuda.synchronize()
        tic = perf_counter_ns()
        execute_sobel()
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