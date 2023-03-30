from PIL import Image
import numpy as np

rgb = [[46, 58, 35],
       [255, 24, 46],
       [255, 24, 205],
       [165, 24, 255],
       [43, 24, 255],
       [251, 206, 177],
       [24, 255, 119],
       [24, 177, 255],
       [225, 255, 24],
       [120, 88, 64]]

image = Image.open('D:\\geometry_figures.jpg', mode='r')
src = np.array(image)

n_sum = src.shape[0] * src.shape[1]

count = [0 * i for i in range(256)]

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

dictionary = {}
counter = 1
for i in range(len(arr) - 2):
    for j in range(len(arr[i]) - 2):
        if arr[i + 1][j + 1] == 1:
            if arr[i][j + 1] != 0:
                if arr[i + 1][j] != 0:
                    if arr[i + 1][j] < arr[i][j + 1]:
                        dictionary[arr[i][j + 1]] = arr[i + 1][j]
                        arr[i + 1][j + 1] = arr[i + 1][j]
                    elif arr[i + 1][j] > arr[i][j + 1]:
                        dictionary[arr[i + 1][j]] = arr[i][j + 1]
                        arr[i + 1][j + 1] = arr[i][j + 1]
                    else:
                        arr[i + 1][j + 1] = arr[i][j + 1]
                else:
                    arr[i + 1][j + 1] = arr[i][j + 1]
            elif arr[i + 1][j] != 0:
                arr[i + 1][j + 1] = arr[i + 1][j]
            else:
                counter += 1
                arr[i + 1][j + 1] = counter
                if counter not in dictionary.keys():
                    dictionary[counter] = 0

for key in dictionary.keys().__reversed__():
    for i in range(len(arr) - 2):
        for j in range(len(arr[i]) - 2):
            if arr[i + 1][j + 1] == key and dictionary[key] != 0:
                arr[i + 1][j + 1] = dictionary[key]

left_numbers = []
for key in dictionary.keys():
    if dictionary[key] == 0:
        left_numbers.append(key)

for i in range(src.shape[0]):
    for j in range(src.shape[1]):
        for k in range(len(left_numbers)):
            if arr[i + 1][j + 1] == left_numbers[k]:
                src[:, :, 0][i][j] = rgb[k][0]
                src[:, :, 1][i][j] = rgb[k][1]
                src[:, :, 2][i][j] = rgb[k][2]

data = Image.fromarray(src)
data.save('assets\\colored_gf.png')
print(counter - 2)


