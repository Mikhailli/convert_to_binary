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


count = [0 * i for i in range(256)]
def build_п_matrix(width: int, height: int):
    matrix = np.zeros(width * height, dtype="uint8")
    matrix = np.reshape(matrix, (height, width))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if i == 0 or i == 1 or j == 0 or j == 1 or j == width - 1 or j == width - 2:
                matrix[i][j] = 1
    return matrix


def build_р_matrix(width: int, height: int):
    matrix = np.zeros(width * height, dtype="uint8")
    matrix = np.reshape(matrix, (height, width))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if j == 0 or j == 1:
                matrix[i][j] = 1
            if i == 0 and j != width - 1 and j != width - 2:
                matrix[i][j] = 1
            if i == 1 and j != width - 1:
                matrix[i][j] = 1
            if i == 2 and (j == width - 1 or j == width - 2 or j == width - 3):
                matrix[i][j] = 1
            if i == int((height - 4) / 2) + 3 and j != width - 1 and j != width - 2:
                matrix[i][j] = 1
            if i == int((height - 4) / 2) + 2 and j != width - 1:
                matrix[i][j] = 1
            if i == int((height - 4) / 2) + 1 and (j == width - 1 or j == width - 2 or j == width - 3):
                matrix[i][j] = 1
            if int((height - 4) / 2) + 1 > i > 2 and (j == width - 1 or j == width - 2):
                matrix[i][j] = 1
    return matrix


def build_г_matrix(width: int, height: int):
    matrix = np.zeros(width * height, dtype="uint8")
    matrix = np.reshape(matrix, (height, width))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if j == 0 or j == 1 or i == 0 or i == 1:
                matrix[i][j] = 1
    return matrix


def build_н_matrix(width: int, height: int):
    matrix = np.zeros(width * height, dtype="uint8")
    matrix = np.reshape(matrix, (height, width))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if j == 0 or j == 1 or j == width - 1 or j == width - 2:
                matrix[i][j] = 1
            if height % 2 == 0:
                if i == height / 2 or i == height / 2 - 1:
                    matrix[i][j] = 1
            else:
                if i == (height + 1) / 2 or i == (height + 1) / 2 - 1 or i == (height + 1) / 2 - 2:
                    matrix[i][j] = 1
    return matrix


def build_т_matrix(width: int, height: int):
    matrix = np.zeros(width * height, dtype="uint8")
    matrix = np.reshape(matrix, (height, width))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if i == 0 or i == 1:
                matrix[i][j] = 1
            if width % 2 == 0:
                if j == width / 2 or j == width / 2 - 1:
                    matrix[i][j] = 1
            else:
                if j == (width + 1) / 2 or j == (width + 1) / 2 - 1 or j == (width + 1) / 2 - 2:
                    matrix[i][j] = 1
    return matrix


def pick_letter(width, height, letter):
    matrix_р = build_р_matrix(width, height)
    matrix_п = build_п_matrix(width, height)
    matrix_н = build_н_matrix(width, height)
    matrix_т = build_т_matrix(width, height)
    matrix_г = build_г_matrix(width, height)
    distance_р = 0
    distance_п = 0
    distance_н = 0
    distance_т = 0
    distance_г = 0
    for i in range(height):
        for j in range(width):
            if letter[i][j] != 0 and matrix_р[i][j] == 0 or (letter[i][j] == 0 and matrix_р[i][j] != 0):
                distance_р += 1
            if letter[i][j] != 0 and matrix_п[i][j] == 0 or (letter[i][j] == 0 and matrix_п[i][j] != 0):
                distance_п += 1
            if letter[i][j] != 0 and matrix_н[i][j] == 0 or (letter[i][j] == 0 and matrix_н[i][j] != 0):
                distance_н += 1
            if letter[i][j] != 0 and matrix_т[i][j] == 0 or (letter[i][j] == 0 and matrix_т[i][j] != 0):
                distance_т += 1
            if letter[i][j] != 0 and matrix_г[i][j] == 0 or (letter[i][j] == 0 and matrix_г[i][j] != 0):
                distance_г += 1
    collection = [distance_г, distance_т, distance_р, distance_п, distance_н]
    if distance_г == min(collection):
        return 'г'
    if distance_т == min(collection):
        return 'т'
    if distance_р == min(collection):
        return 'р'
    if distance_п == min(collection):
        return 'п'
    if distance_н == min(collection):
        return 'н'


image = Image.open('D:\\text.bmp', mode='r')

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

for i in range(len(arr) - 2):
    for j in range(len(arr[i]) - 2):
        if arr[i + 1][j + 1] != 0:
            if arr[i + 1][j - 1] != 0:
                if arr[i + 1][j + 1] > arr[i + 1][j - 1]:
                    dictionary[arr[i + 1][j + 1]] = arr[i + 1][j - 1]

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
min_x = np.full((len(left_numbers)), 2 ** 20, dtype=int)
max_x = np.full((len(left_numbers)), -1, dtype=int)
min_y = np.full((len(left_numbers)), 2 ** 20, dtype=int)
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

list = []

for k in range(len(left_numbers)):
    counter = 0
    letters = np.zeros((max_y[k] - min_y[k] + 1) * (max_x[k] - min_x[k] + 1))
    for i in range(len(arr) - 2):
        for j in range(len(arr[i]) - 2):
            if min_y[k] <= i + 1 <= max_y[k] and min_x[k] <= j + 1 <= max_x[k]:
                letters[counter] = arr[i + 1][j + 1]
                counter += 1
    list.append((np.reshape(letters, (max_y[k] - min_y[k] + 1, max_x[k] - min_x[k] + 1)), (min_x[k], max_y[k] - min_y[k] + 1, max_x[k] - min_x[k] + 1)))

list.sort(key=lambda x: x[1][0])
letters = ''
for letter in list:
    letters += pick_letter(letter[1][2], letter[1][1], letter[0])
print(letters)
# t = build_и_matrix(59, 37)
# print(build_и_matrix(59, 37))