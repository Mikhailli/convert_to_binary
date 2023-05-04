from PIL import Image
import numpy as np

count = [0 * i for i in range(256)]

matrix = [[1, 1, 1],
          [1, 1, 1],
          [1, 1, 1]]


def extension(image_arr):
    for i in range(len(image_arr) - 2):
        for j in range(len(image_arr[i]) - 2):
            if image_arr[i + 1][j + 1] == 1:
                if image_arr[i][j] == 0:
                    image_arr[i][j] = 2
                if image_arr[i][j + 1] == 0:
                    image_arr[i][j + 1] = 2
                if image_arr[i][j + 2] == 0:
                    image_arr[i][j + 2] = 2
                if image_arr[i + 1][j] == 0:
                    image_arr[i + 1][j] = 2
                if image_arr[i + 1][j + 2] == 0:
                    image_arr[i + 1][j + 2] = 2
                if image_arr[i + 2][j] == 0:
                    image_arr[i + 2][j] = 2
                if image_arr[i + 2][j + 1] == 0:
                    image_arr[i + 2][j + 1] = 2
                if image_arr[i + 2][j + 2] == 0:
                    image_arr[i + 2][j + 2] = 2
    for i in range(len(image_arr)):
        for j in range(len(image_arr[i])):
            if image_arr[i][j] == 2:
                image_arr[i][j] = 1
    return image_arr


def erosion(image_arr):
    for i in range(len(image_arr) - 2):
        for j in range(len(image_arr[i]) - 2):
            if image_arr[i + 1][j + 1] != 0\
                    and image_arr[i][j] != 0\
                    and image_arr[i][j + 1] != 0\
                    and image_arr[i][j + 2] != 0\
                    and image_arr[i + 1][j] != 0\
                    and image_arr[i + 1][j + 2] != 0\
                    and image_arr[i + 2][j] != 0\
                    and image_arr[i + 2][j + 1] != 0\
                    and image_arr[i + 2][j + 2] != 0:
                image_arr[i + 1][j + 1] = 2
    for i in range(len(image_arr)):
        for j in range(len(image_arr[i])):
            if image_arr[i][j] == 1:
                image_arr[i][j] = 0
    for i in range(len(image_arr)):
        for j in range(len(image_arr[i])):
            if image_arr[i][j] == 2:
                image_arr[i][j] = 1
    return image_arr


def convert_to_image(image_arr):
    for i in range(len(image_arr) - 2):
        for j in range(len(image_arr[i]) - 2):
            if image_arr[i + 1][j + 1] == 1:
                src[:, :, 0][i][j] = 0
            else:
                src[:, :, 0][i][j] = 255
    src[:, :, 1] = src[:, :, 0]
    src[:, :, 2] = src[:, :, 0]
    data = Image.fromarray(src)
    data.save('assets\\gf.png')


image = Image.open('D:\\morfologic.bmp', mode='r')

src = np.array(image)
n_sum = src.shape[0] * src.shape[1]

src[:, :, 1] = src[:, :, 0]
src[:, :, 2] = src[:, :, 0]

data = Image.fromarray(src)
data.save('assets\\gf.png')

for i in range(src.shape[0]):
    for j in range(src.shape[1]):
        count[src[:, :, 0][i][j]] += 1

in_sum = 0

for t in range(0, 256):
    in_sum += t * count[t]

maxSigma = -1
threshold = 0
in_sum1 = 0
n_sum1 = 0
in_sum2 = 0
n_sum2 = 0

for t in range(0, 255):
    in_sum1 += t * count[t]
    n_sum1 += count[t]
    in_sum2 = in_sum - in_sum1
    n_sum2 = n_sum - n_sum1

    q1 = n_sum1 / n_sum
    m1 = 0
    m2 = 0
    if n_sum1 != 0:
        m1 = in_sum1 / n_sum1
    if n_sum2 != 0:
        m2 = in_sum2 / n_sum2

    sigma = q1 * (1 - q1) * ((m1 - m2) ** 2)

    if sigma > maxSigma:
        maxSigma = sigma
        threshold = t

image = Image.open('assets\\gf.png')
src = np.array(image)
for i in range(src.shape[0]):
    for j in range(src.shape[1]):
        if int(src[:, :, 0][i][j]) <= int(threshold):
            src[:, :, 0][i][j] = 0
        else:
            src[:, :, 0][i][j] = 255
src[:, :, 1] = src[:, :, 0]
src[:, :, 2] = src[:, :, 0]

data = Image.fromarray(src)

count = [0 * i for i in range(256)]

for i in range(src.shape[0]):
    for j in range(src.shape[1]):
        count[src[:, :, 0][i][j]] += 1

data.save('assets\\bw_gf.png')

arr = np.zeros((src.shape[0] + 2, src.shape[1] + 2), dtype="uint16")
for i in range(src.shape[0]):
    for j in range(src.shape[1]):
        if src[:, :, 0][i][j] == 0:
            arr[i + 1][j + 1] = 1
        elif src[:, :, 0][i][j] == 255:
            arr[i + 1][j + 1] = 0


arr = erosion(arr)
arr = erosion(arr)
arr = extension(arr)

convert_to_image(arr)
