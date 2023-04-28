import sys
from colorama import Style

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

image = Image.open('D:\\geometry_figures.bmp', mode='r')
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

square = np.zeros(len(left_numbers), dtype=int)
coord_x_sum = np.zeros(len(left_numbers), dtype=int)
coord_y_sum = np.zeros(len(left_numbers), dtype=int)
min_x = np.full((len(left_numbers)), 2**20, dtype=int)
max_x = np.full((len(left_numbers)), -1, dtype=int)
min_y = np.full((len(left_numbers)), 2**20, dtype=int)
max_y = np.full((len(left_numbers)), -1, dtype=int)

for i in range(src.shape[0]):
    for j in range(src.shape[1]):
        for k in range(len(left_numbers)):
            if arr[i + 1][j + 1] == left_numbers[k]:
                if i + 1 > max_y[k]:
                    max_y[k] = i + 1
                if i + 1 < min_y[k]:
                    min_y[k] = i + 1
                if j + 1 > max_x[k]:
                    max_x[k] = j + 1
                if j + 1 < min_x[k]:
                    min_x[k] = j + 1
                src[:, :, 0][i][j] = rgb[k][0]
                src[:, :, 1][i][j] = rgb[k][1]
                src[:, :, 2][i][j] = rgb[k][2]
                square[k] += 1
                coord_x_sum[k] += j
                coord_y_sum[k] += i


data = Image.fromarray(src)
data.save('assets\\colored_gf.png')
print("Количество объектов равно {}".format(len(left_numbers)))

square_roots_of_2 = np.zeros(len(left_numbers), dtype=int)
perimeter = np.zeros(len(left_numbers), dtype=float)

for k in range(len(left_numbers)):
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if min_y[k] <= i + 2 <= max_y[k] + 3 and min_x[k] <= j + 2 <= max_x[k] + 3:
                if arr[i + 1][j + 1] != left_numbers[k]:
                    counter = 0
                    if arr[i + 1][j] == left_numbers[k]:
                        counter += 1
                    if arr[i][j + 1] == left_numbers[k]:
                        counter += 1
                    if arr[i + 1][j + 2] == left_numbers[k]:
                        counter += 1
                    if arr[i + 2][j + 1] == left_numbers[k]:
                        counter += 1
                    if counter == 2:
                        square_roots_of_2[k] += 1
                    else:
                        perimeter[k] += counter

for i in range(len(perimeter)):
    perimeter[i] += float(square_roots_of_2[i]) * float(2**(1/2))


for i in range(len(coord_y_sum)):
    print("__________________________________________________________________________________")
    if i == 0:
        print("\033[32mКвадрат")
    if i == 1:
        print("\033[32mКруг")
    if i == 2:
        print("\033[32mПравильный шестиугольник")
    if i == 3:
        print("\033[32mРомб")
    if i == 4:
        print("\033[32mТреугольник")
    if i == 5:
        print("\033[32mОвал")
    if i == 6:
        print("\033[32mЗвезда")
    if i == 7:
        print("\033[32mПрямоугольник")
    print(Style.RESET_ALL)
    print("Площадь равна: {}".format(square[i]))
    print("Периметр равен: {}".format(round(perimeter[i], 2)))
    print("Коэффициент округлости равен {}".format(round(perimeter[i] * perimeter[i] / square[i], 2)))
    print("Координаты центра масс: ({};{})".format(round(coord_x_sum[i] / square[i], 2), round(coord_y_sum[i] / square[i], 2)))
    print("__________________________________________________________________________________")
