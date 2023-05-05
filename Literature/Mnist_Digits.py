
import numpy as np
import random as rn
import math
import gzip
import matplotlib.pyplot as plt


class NN:
    '''
    w == weights
    b == bias
    h == activation neuron

    h_i = sum(w_i-1 * h_i-1) + b_i
    '''
    def __init__(self) -> None:
        self.input_size = 1
        self.input_layer: list = []
        self.w0:list[list[float]] = []
        
        self.h1:list = []
        self.w1:list[list[float]] = []
        self.b1:list = []
        self.b1_dud:list = []

        self.h2:list = []
        self.w2:list[list[float]] = []
        self.b2:list = []
        self.b2_dud:list = []


        self.output_layer = []
        self.b3:list = []

        self.meta_output = []
        self.percent_tracker: list = []
        self.guess_dif: list = [0,0,0,0,0,0,0,0,0,0]


    """
    Create a row full of random values [0,1)

    input:  int n

    return: list of size n with random values in it
    """    
    def create_row_rand(self, size:int):
        row = []
        rand = rn
        for i in range(size):
            # if(rand.randint(1,2) % 2 == 0):
            #     row.append(-1*rand.random())
            # else:
            row.append(rand.random())
        return row


    '''
    Initializ a layer to the NN
    input:  list layer
            int rows
            int row

    return list[list[float]]
    '''
    def initialize_layer(self, rows: int, row: int, type: str= "") -> list:
        mat = []

        if rows < 1:
            rows = 1
        # if dud layer only fill with 0's
        if type == "dud layer":
            for row_idx in range(rows):
                temp = []
                for col_idx in range(row):
                    temp.append(1)
                mat.append(temp)
            mat = np.array(mat)   
        else:
            for row_idx in range(rows):
                mat.append(self.create_row_rand(row))

        return mat
    
    '''
    Create the NN
    input:  int image_height
            int image_lenght
            int hidden_layers
            int hidden_layers_size
            int output_size

    return:
    '''
    def initialize_NN(self, image_height: int, image_length: int,
                            hidden_layers: int, hidden_layers_size: int,
                            output_size: int) -> None: 
        self.input_size = image_height * image_length

        if hidden_layers_size < 1:
            hidden_layers_size = 1 
        
        self.input_layer = self.initialize_layer(rows=1, row=self.input_size)
        self.w0 = self.initialize_layer(rows=hidden_layers_size, row=len(self.input_layer[0]))

        self.h1 = self.initialize_layer(rows=1, row=hidden_layers_size)
        self.b1 = self.initialize_layer(rows=1, row=hidden_layers_size)
        self.w1 = self.initialize_layer(rows=hidden_layers_size, row=hidden_layers_size)

        self.h2 = self.initialize_layer(rows=1, row=hidden_layers_size)
        self.b2 = self.initialize_layer(rows=1, row=hidden_layers_size)
        self.w2 = self.initialize_layer(rows=output_size, row=hidden_layers_size)

        self.output_layer = self.initialize_layer(rows=1, row=output_size)
        # not used so doesnt matter
        self.b3 = self.initialize_layer(rows=1, row=output_size, type="dud layer")
        self.b2_dud = self.initialize_layer(rows=1, row=hidden_layers_size, type="dud layer")
        self.b1_dud = self.initialize_layer(rows=1, row=hidden_layers_size, type="dud layer")

        self.w0 = np.transpose(np.multiply(np.transpose(self.w0), [0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001]))
        return
    
    '''
    sig(x) = 1/(1 + e^(-x))
    input:  x
    output: sig(x)
    '''
    def sigmoid(self,x:float):
        return 1/(1 + pow(math.e,-x))
    
    '''(
    sig^(-1)(x) = sig(x)*(1 - sig(x))
    '''
    def sigmoid_derivative(self, x:float):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    '''
    Given layer i and i-1 -> run a feed foward algo
    h_i = sigmoid(sum(w_i-1 * h_i-1) + b_i)

    input:  list layer_i_minus_1
            list weights_i_minus_1
            list layer_i_bias

    output: list layer_i updated
    '''
    def feed_foward_layer(self, layer_i_minus_1:list, weights_i_minus_1:list[list], 
                        layer_i_bias:list):
                        
        # x =np.matmul(layer_i_minus_1, np.transpose(weights_i_minus_1))
        x = np.matmul(layer_i_minus_1, np.transpose(weights_i_minus_1))
        activation = np.add(x, layer_i_bias)
        
        for idx in range(len(activation[0])):
            activation[0][idx] = self.sigmoid(activation[0][idx])
        
        return activation


    '''
    Feed the network foward 1 layer at a time
    
    input:  
    
    return: int nn_guess
    '''
    def feed_foward(self, input_values: float):

        for idx in range(len(self.input_layer[0])):   
            self.input_layer[0][idx] = (input_values[idx]/255)
        
        if True:
            pass

        self.h1 = self.feed_foward_layer(layer_i_minus_1=self.input_layer, weights_i_minus_1=self.w0, 
                                layer_i_bias=self.b1)

        self.h2 = self.feed_foward_layer(layer_i_minus_1=self.h1, weights_i_minus_1=self.w1, 
                                layer_i_bias=self.b2)
        
        self.output_layer = self.feed_foward_layer(layer_i_minus_1=self.h2, weights_i_minus_1=self.w2, 
                                layer_i_bias=self.b3)

        return self.output_layer


    # create a copy of something :/
    def copy(self, x):
        return x

    '''fuck you'''
    def backprop(self, answer: list, learning_rate: float):

        # self.input_layer: list = []
        # self.w0:list[list[float]] = []
        
        # self.h1:list = []
        # self.w1:list[list[float]] = []
        # self.b1:list = []

        # self.h2:list = []
        # self.w2:list[list[float]] = []
        # self.b2:list = []

        # self.output_layer = []
        # self.b3:list = []sig_prime_output_layer


        sig_prime_output_layer = []
        new_output_layer = []
        for i in range(len(answer)):
            sig_prime_output_layer.append(self.sigmoid_derivative(answer[i]))
            new_output_layer.append(answer[i])
        
        sig_prime_output_layer = [sig_prime_output_layer]
        new_output_layer = [new_output_layer]
        x = np.transpose(np.multiply(new_output_layer, sig_prime_output_layer))
        delta_w2 = np.matmul(x, self.h2)*learning_rate
        delta_h2 = np.matmul(np.transpose(x),  self.w2)*learning_rate
        delta_b2 = np.multiply(delta_h2, self.sigmoid_derivative(self.h2))*learning_rate

        x =  np.transpose(np.multiply(delta_h2, self.sigmoid_derivative(self.h2)))
        delta_w1 = np.matmul(x, self.h1)
        delta_h1 = np.matmul(np.transpose(x), self.w1)
        delta_b1 = np.multiply(delta_h1, self.sigmoid_derivative(self.h1))
        

        x =  np.transpose(np.multiply(delta_h1, self.sigmoid_derivative(self.h1)))
        # main_path = np.matmul(main_path, np.transpose(self.sigmoid_derivative(self.h1)))
        # y = np.matmul(np.transpose(self.sigmoid_derivative(self.h1)), delta_h1)
        delta_w0 = np.matmul(x, self.input_layer)

        # self.w2 = np.add(self.w2, np.transpose(delta_w2))
        # self.b2 = np.add(self.b2, np.transpose(delta_b2))
        # self.w1 = np.add(self.w1, np.transpose(delta_w1))
        # self.b1 = np.add(self.b1, delta_b1)
        # self.w0 = np.add(self.w0, np.transpose(delta_w0))

        self.w2 = np.subtract(self.w2, delta_w2)
        self.b2 = np.subtract(self.b2, delta_b2)
        self.w1 = np.subtract(self.w1, delta_w1)
        self.b1 = np.subtract(self.b1, delta_b1)
        self.w0 = np.subtract(self.w0, delta_w0)

        pass

    '''
    grade the NN output agains the correct answer
    input:  list guess
            list answer
    return: list answer - list
    '''
    def grade(self, guess: list, answer:list):
        return pow(np.subtract(guess[0] , answer),2)   #I know this looks backwards

    
    '''
    Gets images and labels from the training set of a set batch size
    input:  idx_start
            idx_end
    return: [images, labels] 
    '''
    def get_training_images_labels(self, idx_start: int, idx_end: int):

        f = gzip.open('train-images-idx3-ubyte.gz','r')
        image_size = 28
        num_images = idx_end - idx_start
        f.read(16)
        f.seek(image_size * image_size * idx_start + 16)
        buf = f.read(image_size * image_size * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, image_size, image_size, 1)
        f.close()


        # extract a single picture
        # image = np.asarray(data[0]).squeeze()
        
        # plt.imshow(image)
        # plt.show()



        f = gzip.open('train-labels-idx1-ubyte.gz','r')
        f.read(8)
        f.seek(idx_start + 8)
        labels = []
        for i in range(idx_start, idx_end):   
            buf = f.read(1)
            label = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
            labels.append(label)
        f.close()

        # image = np.asarray(data[0]).squeeze()
        # print(labels[0], )
        # plt.imshow(image)
        # plt.show()

        return data, labels


    '''
    Runs a batch through the NN and grades it along the way.
    Grades are summed and then taken to backprop
    input:  idx_start
            idx_end
    retrun: 
    '''
    def run_batch(self, idx_start: int, idx_end: int, batch_count: int, epoch_count: int):
        images, labels = self.get_training_images_labels(idx_start=idx_start, idx_end=idx_end)

        number_correct = 0
        summed_answers = []
        for i in range(len(self.output_layer)):
                summed_answers.append(0)
        last_guess = [1,1,1,1,1,1,1,1,1,1]
        idx = 0
        track_outcomes = [0,0,0,0,0,0,0,0,0,0]
        while idx < len(labels):
            image = np.asarray(images[idx]).squeeze().flatten()
            guess = self.feed_foward(input_values=image)
            
            self.guess_dif = self.guess_dif + (last_guess - guess)
            answer = []
            for i in range(len(self.output_layer[0])):
                answer.append(0)
            answer[labels[idx][0]] = 1
            track_outcomes[labels[idx][0]] += 1

            # mage = np.asarray(images[idx]).squeeze()
            # print(labels[idx][0], "     ",  np.argmax(guess))
            # plt.imshow(mage)
            # plt.show()

            to_be_evaluated = self.grade(guess=guess, answer=answer)
            summed_answers = np.add(summed_answers, to_be_evaluated)

            
            if (np.argmax(guess) == np.argmax(answer)):
                number_correct += 1

            idx += 1
        self.percent_tracker.append(number_correct/(idx_end-idx_start))
        print("Percent correct: ", number_correct/(idx_end-idx_start), "   From Batch: ", batch_count, "  Epoch:  ", epoch_count)
        print("number correctr: ", number_correct,"  start:    ",idx_start, "    end: ", idx_end )
        return np.divide(summed_answers, track_outcomes)
        # return summed_answers/(idx_end - idx_start)


    def epoch(self, number_of_epochs: int, batch_size: int, learning_rate: float,epoch_size=60000):
        
        total_images = epoch_size
        idx = 0
        batches = []
        while idx < total_images:
            idx += batch_size
            batches.append(idx)
        
        self.meta_output = [[],[],[],[],[],[],[],[],[],[]] 
        for epoch_idx in range(number_of_epochs):
            start = 0
            count = 0
            for batch in batches:
                count +=1
                summed_answers = self.run_batch(idx_start=start, idx_end=batch, batch_count=count, epoch_count=epoch_idx)
                print("Summed answers: ", summed_answers)
                self.backprop(answer=summed_answers, learning_rate=learning_rate)
                for i in range(len(self.meta_output)):
                    self.meta_output[i].append(summed_answers[i])
                start = batch
                
             


def main():
    mnits_digits = NN()
    mnits_digits.initialize_NN(image_height=28, image_length=28, 
                                    hidden_layers=2, hidden_layers_size=8,
                                    output_size=10)
    # guess = mnits_digits.feed_foward()
    # mnits_digits.get_training_images_labels(idx_start=56,idx_end=300)
    epochs = 10
    batch_size = 100
    learning_rate = 0.1
    mnits_digits.epoch(number_of_epochs=epochs,batch_size=batch_size, learning_rate=learning_rate, epoch_size=10000)

    fig = plt.figure()
    x = []
    for i in range(len(mnits_digits.percent_tracker)):
        x.append(i)
    plt.plot(x, mnits_digits.percent_tracker)
    plt.plot(x, mnits_digits.meta_output[0])
    plt.plot(x, mnits_digits.meta_output[1])
    plt.plot(x, mnits_digits.meta_output[2])
    plt.plot(x, mnits_digits.meta_output[3])
    plt.plot(x, mnits_digits.meta_output[4])
    plt.plot(x, mnits_digits.meta_output[5])
    plt.plot(x, mnits_digits.meta_output[6])
    plt.plot(x, mnits_digits.meta_output[7])
    plt.plot(x, mnits_digits.meta_output[8])
    plt.plot(x, mnits_digits.meta_output[9])
    # plt.imshow(plot)
    plt.show()


    pass

if __name__ == "__main__":
    main()