import time
import numpy
from dash import dash_table, State
from PIL import Image
import numpy as np
import dash
from dash import dcc, Output, Input
from dash import html
from dash.exceptions import PreventUpdate

# image = Image.open('assets\\black_and_white.jpg')
# pixels = np.asarray(image)
#
# counter1 = 0
# counter3 = 0
#
#
# for i in range(len(pixels[:, :, 0]) - 1):
#     for j in range(len(pixels[:, :, 0][i]) - 1):
#         if int(pixels[:, :, 0][i, j]) > 10 and int(pixels[:, :, 0][i + 1, j]) <= 10 and int(pixels[:, :, 0][i, j + 1]) <= 10 and int(pixels[:, :, 0][i + 1, j + 1]) <= 10 \
#                 or int(pixels[:, :, 0][i, j]) <= 10 and int(pixels[:, :, 0][i + 1, j]) > 10 and int(pixels[:, :, 0][i, j + 1]) <= 10 and int(pixels[:, :, 0][i + 1, j + 1]) <= 10 \
#                 or int(pixels[:, :, 0][i, j]) <= 10 and int(pixels[:, :, 0][i + 1, j]) <= 10 and int(pixels[:, :, 0][i, j + 1]) > 10 and int(pixels[:, :, 0][i + 1, j + 1]) <= 10 \
#                 or int(pixels[:, :, 0][i, j]) <= 10 and int(pixels[:, :, 0][i + 1, j]) <= 10 and int(pixels[:, :, 0][i, j + 1]) <= 10 and int(pixels[:, :, 0][i + 1, j + 1]) > 10:
#             counter1 += 1
#
#         if int(pixels[:, :, 0][i, j]) > 10 and int(pixels[:, :, 0][i + 1, j]) > 10 and int(pixels[:, :, 0][i, j + 1]) > 10 and int(pixels[:, :, 0][i + 1, j + 1]) <= 10 \
#                 or int(pixels[:, :, 0][i, j]) <= 10 and int(pixels[:, :, 0][i + 1, j]) > 10 and int(pixels[:, :, 0][i, j + 1]) > 10 and int(pixels[:, :, 0][i + 1, j + 1]) > 10 \
#                 or int(pixels[:, :, 0][i, j]) > 10 and int(pixels[:, :, 0][i + 1, j]) <= 10 and int(pixels[:, :, 0][i, j + 1]) > 10 and int(pixels[:, :, 0][i + 1, j + 1]) > 10 \
#                 or int(pixels[:, :, 0][i, j]) > 10 and int(pixels[:, :, 0][i + 1, j]) > 10 and int(pixels[:, :, 0][i, j + 1]) <= 10 and int(pixels[:, :, 0][i + 1, j + 1]) > 10:
#             counter3 += 1


# print("Количество белых пятен: {}".format(int((counter1 - counter3) / 4)))
from plotly.express import pd

image = Image.open('D:\\building.png', mode='r')
src = np.array(image)

n_sum = src.shape[0] * src.shape[1]

bites = [1 * i for i in range(256)]
count = [0 * i for i in range(256)]

src[:, :, 1] = src[:, :, 0]
src[:, :, 2] = src[:, :, 0]

data = Image.fromarray(src)
data.save('assets\\bw.png')

for i in range(src.shape[0]):
    for j in range(src.shape[1]):
        count[src[:, :, 0][i][j]] += 1


# in_sum = 0
#
# for t in range(0, 256):
#     in_sum += t * count[t]
#
# maxSigma = -1
# threshold = 0
# in_sum1 = 0
# n_sum1 = 0
# in_sum2 = 0
# n_sum2 = 0
#
# for t in range(0, 255):
#     in_sum1 += t * count[t]
#     n_sum1 += count[t]
#     in_sum2 = in_sum - in_sum1
#     n_sum2 = n_sum - n_sum1
#
#     q1 = n_sum1 / n_sum
#
#     m1 = in_sum1 / n_sum1
#     m2 = in_sum2 / n_sum2
#
#     sigma = q1 * (1 - q1) * ((m1 - m2) ** 2)
#
#     if sigma > maxSigma:
#         maxSigma = sigma
#         threshold = t
#
# print(threshold)
#
# image1 = Image.open('assets\\bw.png')
# src1 = np.array(image1)
# for i in range(src.shape[0]):
#     for j in range(src.shape[1]):
#         if int(src[:, :, 0][i][j]) <= int(threshold):
#             src1[:, :, 0][i][j] = 0
#         else:
#             src1[:, :, 0][i][j] = 255
# src1[:, :, 1] = src1[:, :, 0]
# src1[:, :, 2] = src1[:, :, 0]
#
# data1 = Image.fromarray(src1)
#
# data1.save('assets\\bw_change.png')

params = [
    'FirstColumn', 'SecondColumn', 'ThirdColumn'
]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    # html.H1(children='Hello Dash'),
    #
    # html.Div(children='''
    #     Dash: A web application framework for Python.
    # '''),
    #
    # dcc.Graph(
    #     id='example-graph',
    #     figure={
    #         'data': [
    #             {'x': bites, 'y': count, 'type': 'bar', 'name': 'SF'},
    #         ],
    #         'layout': {
    #             'title': 'Dash Data Visualization'
    #         }
    #     }
    # ),

    html.Div([
        html.Img(width='50%', id='initial_image', src='assets\\bw.png'),

    ], style={'textAlign': 'center'}),

    # html.Div(children='''Оптимальное значение порога: {}
    # '''.format(threshold), style={'textAlign': 'center'}),
    #
    # html.Div([
    #     html.Img(width='50%', id='image', src='assets\\bw_change.png'),
    #
    # ], style={'textAlign': 'center'}),

    # html.Label('Slider'),
    # dcc.Slider(
    #     0, 255, 5,
    #     id='my-slider',
    # ),
html.Div([
    dash_table.DataTable(
        id='table-editing-simple',
        columns=(
            [{'id': 'FirstColumn', 'name': ''}] +
            [{'id': 'SecondColumn', 'name': ''}] +
            [{'id': 'ThirdColumn', 'name': ''}]
        ),
        data=[
            dict(**{param: 0 for param in params})
            for i in range(1, 4)
        ],
        style_data={
        'whiteSpace': 'normal',
        'height': '100px',
        'width': '100px',
        'textAlign': 'center',
        'margin-left': '100px',
        'margin-right': 'auto'
        },
        editable=True,
        fill_width=False,
        style_header = {'display': 'none'}
    ),
    #dcc.Graph(id='table-editing-simple-output')
], style={
            'textAlign': 'center',
            "margin-left":"40%"
        }),
    html.Div([
        html.Button('Click here to see the content', id='show-secret'),
        html.Div(id='body-div')
    ])
])


def is_pixel_on_left_border(pixel_width):
    if pixel_width == 0:
        return True
    return False


def is_pixel_on_top_border(pixel_height):
    if pixel_height == 0:
        return True
    return False


def is_pixel_on_right_border(pixel_width, width):
    if pixel_width == width - 1:
        return True
    return False


def is_pixel_on_bottom_border(pixel_height, height):
    if pixel_height == height - 1:
        return True
    return False


def is_pixel_on_left_top_corner(pixel_width, pixel_height):
    if is_pixel_on_left_border(pixel_width) and is_pixel_on_top_border(pixel_height):
        return True
    return False


def is_pixel_on_right_top_corner(pixel_width, width, pixel_height):
    if is_pixel_on_right_border(pixel_width, width) and is_pixel_on_top_border(pixel_height):
        return True
    return False


def is_pixel_on_right_bottom_corner(pixel_width, width, pixel_height, height):
    if is_pixel_on_right_border(pixel_width, width) and is_pixel_on_bottom_border(pixel_height, height):
        return True
    return False


def is_pixel_on_left_bottom_corner(pixel_width, pixel_height, height):
    if is_pixel_on_left_border(pixel_width) and is_pixel_on_bottom_border(pixel_height, height):
        return True
    return False


clicks = None

@app.callback(
    Output(component_id='initial_image', component_property='src'),
    Input(component_id='show-secret', component_property='n_clicks'),
    Input('table-editing-simple', 'data'),
    Input('table-editing-simple', 'columns'),
    prevent_initial_call=True
)
def update_output_div(n_clicks, rows, columns):
    var = dash.callback_context
    if n_clicks is None:
        raise PreventUpdate
    tic = time.perf_counter()
    mask = []
    for row in rows:
        for pair in row:
            value = row.get(pair)
            mask.append(int(value))
    new_pixels = src[:, :, 0]

    array = np.zeros((src.shape[0] + 2, src.shape[1] + 2), dtype="uint8")
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            array[i + 1][j + 1] = src[:, :, 0][i][j]

    mask_sum = 0
    for number in mask:
        mask_sum += int(number)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if mask_sum == 0:
                new_pixels[i][j] = numpy.uint8(0)
            else:
                temp = int(array[i][j] * mask[0]) + int(array[i][j + 1] * mask[1]) + int(array[i][j + 2] * mask[2]) + \
                    int(array[i + 1][1] * mask[3]) + int(array[i + 1][j + 1] * mask[4]) + int(array[i + 1][j + 2] * mask[5]) + \
                    int(array[i + 2][j] * mask[6]) + int(array[i + 2][j + 1] * mask[7]) + int(array[i + 2][j + 2] * mask[8])
                new_pixels[i][j] = numpy.uint8(temp / mask_sum)
    src[:, :, 0] = new_pixels
    src[:, :, 1] = new_pixels
    src[:, :, 2] = new_pixels
    new_data = Image.fromarray(src)

    new_data.save('assets\\bw.png')
    toc = time.perf_counter()
    print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")
    return app.get_asset_url('bw.png')
def update_output(value):
    image1 = Image.open('assets\\bw.png')
    src1 = np.array(image1)
    if value is None:
        return app.get_asset_url('bw.png')
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if int(src[:, :, 0][i][j]) <= int(value):
                src1[:, :, 0][i][j] = 0
            else:
                src1[:, :, 0][i][j] = 255
    src1[:, :, 1] = src1[:, :, 0]
    src1[:, :, 2] = src1[:, :, 0]

    data1 = Image.fromarray(src1)

    data1.save('assets\\bw.png')
    return app.get_asset_url('bw.png')


if __name__ == '__main__':
    app.run_server(debug=True)
