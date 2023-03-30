import time
import numpy
from PIL import Image
import numpy as np
import dash
from dash import dcc, Output, Input
from dash import html
from dash.exceptions import PreventUpdate

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

    m1 = in_sum1 / n_sum1
    m2 = in_sum2 / n_sum2

    sigma = q1 * (1 - q1) * ((m1 - m2) ** 2)

    if sigma > maxSigma:
        maxSigma = sigma
        threshold = t

print(threshold)

image1 = Image.open('assets\\bw.png')
src1 = np.array(image1)
for i in range(src.shape[0]):
    for j in range(src.shape[1]):
        if int(src[:, :, 0][i][j]) <= int(threshold):
            src1[:, :, 0][i][j] = 0
        else:
            src1[:, :, 0][i][j] = 255
src1[:, :, 1] = src1[:, :, 0]
src1[:, :, 2] = src1[:, :, 0]

data1 = Image.fromarray(src1)

data1.save('assets\\bw_change.png')


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.Div([
        html.Img(width='50%', id='initial_image', src='assets\\bw.png'),

    ], style={'textAlign': 'center'}),

])


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
