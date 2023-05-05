
'''
TODO: Gather data for expariment 4 -> humans look at 2 blocks and asked if it is white, black, or grey
TODO: Implement the S4 expariment
'''






import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display
import sys

ei = ""
cb = ""
pvmax = ""
pvmin = ""
ep = ""
ds = ""
if (input("Do you wish to use the defaults? [Y/N]") == "N"):
    ei = input("Which Expariment do you wish to do? int [1 -> 4]")
    cb = input("what do you want as a color bias? int [0 -> inf]")
    pvmax = input("what is the maximum pixel value?")
    pvmin = input("what is the minimum pixel value?")
    ep = input("How many epochs?")
    ds = input("how many data points do you wish to train on?")

experiment = int(ei) if ei != "" else 1 
color_bias = int(cb) if cb != "" else 150
pixel_val_max = int(pvmax) if pvmax != "" else 255
pixel_val_min = int(pvmin) if pvmax != "" else 0

epochs = int(ep) if ep != "" else 100
data_size = int(ds) if ds != "" else 1000


'''
intersting setups:

1.)
experiment      = 1
color_bias      = 150
pixel_val_max   = 255
pixel_val_min   = 0
epochs          = 100
data_size       = 1000

2.)
experiment      = 3
color_bias      = 170
pixel_val_max   = 255
pixel_val_min   = 0
epochs          = 100
data_size       = 1000



'''




# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Define the neural network model
model = Sequential([
    Dense(3, activation='sigmoid', input_shape=(2,)),
    Dense(3, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.01), loss='mse', metrics=['accuracy'])

# Generate training data
X = np.random.randint(pixel_val_min, pixel_val_max, (data_size, 2))

'''Math defined, black, white, or grey'''
def S1(X, n, input_tracker):
    y = np.zeros((X.shape[0], 3))
    answer_tracker = [0,0,0]
    for i, (x1, x2) in enumerate(X):
        answer_tracker = [0,0,0]
        if (x1 + x2) < (2*pixel_val_min + n):
            answer_tracker[0] = 1
            y[i, 0] = 1
        elif (x1 + x2) > (2*pixel_val_max - n):
            answer_tracker[1] = 1
            y[i, 1] = 1
        else:
            answer_tracker[2] = 1
            y[i, 2] = 1
        input_tracker = np.add(answer_tracker,input_tracker) 
    return y, input_tracker


'''More restrictive Math defined, black, white, or grey'''
def S2(X, n, input_tracker):
    y = np.zeros((X.shape[0], 3))
    answer_tracker = [0,0,0]
    for i, (x1, x2) in enumerate(X):
        answer_tracker = [0,0,0]
        if x1 > x2 and x1 < (pixel_val_min+n):
            answer_tracker[0] = 1
            y[i, 0] = 1
        elif x1 < x2 and x2 > (pixel_val_max - n):
            answer_tracker[1] = 1
            y[i, 1] = 1
        else:
            answer_tracker[2] = 1
            y[i, 2] = 1
        input_tracker = np.add(answer_tracker,input_tracker) 
    return y, input_tracker

'''Math defined, black, white, or grey BUT every third answer is neither'''
def S3(X, n, input_tracker):
    y = np.zeros((X.shape[0], 3))
    answer_tracker = [0,0,0]
    counter = 0
    for i, (x1, x2) in enumerate(X):
        counter += 1
        answer_tracker = [0,0,0]
        if x1 + x2 < n and counter !=3:
            answer_tracker[0] = 1
            y[i, 0] = 1
        elif x1 + x2 > n and counter != 3:
            answer_tracker[1] = 1
            y[i, 1] = 1
        else:
            answer_tracker[2] = 1
            y[i, 2] = 1
            counter = 0
        input_tracker = np.add(answer_tracker,input_tracker) 
    return y, input_tracker

# TODO: Gather the human defined data and finish this function
'''Human defined, black, white, or grey'''
def S4(X, n, input_tracker):
    human_data, human_answers = __S4()
    y = np.zeros((X.shape[0], 3))
    answer_tracker = [0,0,0]
    for i in human_answers:
        answer_tracker = [0,0,0]
        if i[0] == 1:
            answer_tracker[0] = 1
            y[i, 0] = 1
        elif i[1] == 1:
            answer_tracker[1] = 1
            y[i, 1] = 1
        else:
            answer_tracker[2] = 1
            y[i, 2] = 1
        input_tracker = np.add(answer_tracker,input_tracker) 
    return y, input_tracker

def __S4():
    file_data_list = []
    with open("data.txt") as f:
        file_data_list = f.readlines()
    data = []
    answers = []
    for line in file_data_list:
        input1, input2, answer1,answer2, answer3 = line.split(",")
        data.append([int(input1), int(input2)])
        answers.append([int(answer1), int(answer2), int(answer3)])
    return data, answers


def pick_experiment(expar_number, X, color_bias, sum_of_inputs):
    y = 0
    if expar_number == 1:
        y, sum_of_inputs = S1(X, color_bias, sum_of_inputs)
    elif expar_number == 2:
        y, sum_of_inputs = S2(X, color_bias, sum_of_inputs)
    elif expar_number == 3:
        y, sum_of_inputs = S3(X, color_bias, sum_of_inputs)
    elif expar_number == 4:
        y, sum_of_inputs = S4(X, color_bias, sum_of_inputs)
    return y, sum_of_inputs

sum_of_inputs = [0,0,0]
# human_data, human_answers = 
y, sum_of_inputs = pick_experiment(expar_number=experiment,X=X, color_bias=color_bias, sum_of_inputs=sum_of_inputs)
# Train the model
model.fit(X, y, epochs=epochs, verbose=2)


def create_plot(weight_updates, bias_updates):
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        weights[0] += weight_updates[i]
        weights[1] += bias_updates[i]
        layer.set_weights(weights)

    x_vals = np.linspace(pixel_val_min, pixel_val_max, 50)
    y_vals = np.linspace(pixel_val_min, pixel_val_max, 50)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    input_grid = np.column_stack((X_grid.ravel(), Y_grid.ravel()))
    output_grid = model.predict(input_grid)

    fig = make_subplots(rows=1, cols=4, specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}, {'type': 'scatter3d'}]])

    trace1 = go.Surface(x=X_grid, y=Y_grid, z=output_grid[:, 0].reshape(X_grid.shape), showscale=False,
                        hovertemplate="Input Neuron 1 : %{x} <br>Input Neuron 2 : %{y} <br>Output Neuron 1: %{z} <br> Output Neuron 2: %{customdata[0]} <br>Output Neuron 3: %{customdata[1]}<extra></extra>",
                        customdata=np.column_stack((output_grid[:, 1], output_grid[:, 2])).reshape(X_grid.shape[0], X_grid.shape[1], 2))
                        # customdata=np.column_stack((output_grid[:, 1], output_grid[:, 2], np.ones_like(output_grid[:, 0]) * 42, np.ones_like(output_grid[:, 0]) * 99)).reshape(X_grid.shape[0], X_grid.shape[1], 4))

        
    trace2 = go.Surface(x=X_grid, y=Y_grid, z=output_grid[:, 1].reshape(X_grid.shape), showscale=False,
                        hovertemplate="Input Neuron 1: %{x}, Input Neuron 2: %{y}<br>Output Neuron 1: %{customdata[0]}<br>Output Neuron 2: %{z}<br>Output Neuron 3: %{customdata[1]}<extra></extra>",
                        customdata=np.column_stack((output_grid[:, 0], output_grid[:, 2])).reshape(X_grid.shape[0], X_grid.shape[1], 2))
    trace3 = go.Surface(x=X_grid, y=Y_grid, z=output_grid[:, 2].reshape(X_grid.shape), showscale=False,
                        hovertemplate="Input Neuron 1: %{x}, Input Neuron 2: %{y}<br>Output Neuron 1: %{customdata[0]}<br>Output Neuron 2: %{customdata[1]}<br>Output Neuron 3: %{z}<extra></extra>",
                        customdata=np.column_stack((output_grid[:, 0], output_grid[:, 1])).reshape(X_grid.shape[0], X_grid.shape[1], 2))
    
    trace4 = go.Scatter3d(x=input_grid[:, 0], y=input_grid[:, 1], z=output_grid[:, 0], mode='markers', marker=dict(size=3, color='red'), name='Output Neuron 1',
                        hovertemplate="Input Neuron 1: %{x}<br> Input Neuron 2: %{y}<br> Output Neuron 1: %{z}")



    trace5 = go.Scatter3d(x=input_grid[:, 0], y=input_grid[:, 1], z=output_grid[:, 1], mode='markers', marker=dict(size=3, color='green'), name='Output Neuron 2',
                        hovertemplate="Input Neuron 1: %{x}<br> Input Neuron 2: %{y}<br> Output Neuron 2: %{z}")
    
    trace6 = go.Scatter3d(x=input_grid[:, 0], y=input_grid[:, 1], z=output_grid[:, 2], mode='markers', marker=dict(size=3, color='blue'), name='Output Neuron 3',
                        hovertemplate="Input Neuron 1: %{x}<br> Input Neuron 2: %{y}<br> Output Neuron 2: %{z}")


    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=2)
    fig.add_trace(trace3, row=1, col=3)
    
    fig.add_trace(trace4, row=1, col=4)
    fig.add_trace(trace5, row=1, col=4)
    fig.add_trace(trace6, row=1, col=4)
    
    min_z = min(output_grid.min(), output_grid.min(), output_grid.min())
    max_z = min(max(output_grid.max(), output_grid.max(), output_grid.max()),1) 
    
    fig.update_layout(hovermode="x unified", hoverdistance=50,
                      scene=dict(xaxis_title="Input Neuron 1", yaxis_title="Input Neuron 2", zaxis_title="Output Neuron 1", zaxis_range=[min_z, max_z]),
                      scene2=dict(xaxis_title="Input Neuron 1", yaxis_title="Input Neuron 2", zaxis_title="Output Neuron 2",  zaxis_range=[min_z, max_z]),
                      scene3=dict(xaxis_title="Input Neuron 1", yaxis_title="Input Neuron 2", zaxis_title="Output Neuron 3", zaxis_range=[min_z, max_z]),
                      scene4=dict(xaxis_title="Input Neuron 1", yaxis_title="Input Neuron 2", zaxis_title="Output Neurons"))

    return fig

def plot(weight_update, bias_update):
    weight_updates = [np.ones((2, 3)) * weight_update, np.ones((3, 3)) * weight_update]
    bias_updates = [np.ones(3) * bias_update, np.ones(3) * bias_update]
    fig = create_plot(weight_updates, bias_updates)
    fig.show()

weight_slider = widgets.FloatSlider(min=-1, max=1, step=0.1, value=0, description="Weight Update")
bias_slider = widgets.FloatSlider(min=-1, max=1, step=0.1, value=0, description="Bias Update")

print("These were ",sum_of_inputs, " of each example")
widgets.interact(plot, weight_update=weight_slider, bias_update=bias_slider)