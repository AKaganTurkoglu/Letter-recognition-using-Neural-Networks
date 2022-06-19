import numpy as np

#Neuron class. All of the larning will be performed in this class.
class Neuron:
    name = "empty" #Letter to thought
    epoch_limit = 0 #Epoch limit
    learning_factor = 0.001 #Learning factor
    threshold = 0.001 #Threshold value
    bias = 0.0001 #Bias value
    weights = [] #Weights of each cell

    def set_neuron(self, name, epoch_limit, learning_factor, threshold, bias, input_length): #set instances of class
        self.name = name
        self.epoch_limit = epoch_limit
        self.learning_factor = learning_factor
        self.threshold = threshold
        self.bias = bias
        self.weights = np.zeros(input_length)
        return

    def total_calculate(self, input_data): #the result. Multiplies weights with input value, both in the same index.
        total = self.bias
        for i in range(0, len(input_data)):
            total = total + self.weights[i] * input_data[i]
        return total

    def is_it_completed(self, value_calculated, goal): #Checks if the goal has been reached. Higher or lower than threshold.
        if goal == 1:
            if value_calculated >= self.threshold:
                return True
            else:
                return False
        if goal == -1:
            if value_calculated >= self.threshold:
                return False
            else:
                return True

    def neuron_match(self, input_data): # Returns numeric true or false. Similar to is_it_completed.
        value = self.total_calculate(input_data)
        if value > self.threshold:
            return 1
        else:
            return -1

    def teach_neuron(self, input_data, target_class): #learning function.
        completed = False #This will be used to check if goal has been reached.
        i = 0 #epoch tracker

        while completed is False and i < self.epoch_limit:
            for j in range(0, len(self.weights)): #travel all weights
                self.weights[j] = self.weights[j] + (self.learning_factor *
                                                     (target_class - self.neuron_match(input_data)) *
                                                     input_data[j])
                #delta rule. weight updated both positive and negative feedback.
                #Basically, a new value calculated with the following formula
                #learning factor times desired output (such as 1 or -1) minus goal result (such as 1 orr -1) times input value of the same index.
            self.bias = self.bias + self.learning_factor * (target_class - self.neuron_match(input_data))
            #bias is also updated. As descripted as in delta rule.
            sum_of_array = self.total_calculate(input_data)
            #all weights multiplied by input value of same indexed value.
            completed = self.is_it_completed(sum_of_array, target_class)
            #checked if threshold is reached.
            i = i + 1

    def test_neuron(self, test_data): #After learning, testing with another input
        result = self.total_calculate(test_data) #weights times input
        accuracy = (result/self.threshold) # accuracy defined by this calculation. Only to give ideas and find the most similar letter.
        return self.name , round(accuracy,2) #returns the letter and accuracy - rounded to two decimals

#input values. All are actually 7x9 matrices. 1 means a dot at the index, -1 means empty.
#There are three fonts for all classes.
#The order of matrix is A1,B1,C1,D1,E1,J1,K1,A2,B2,...
input_matrix = [
    [-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,1,1,1],
    [1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1],
    [-1,-1,1,1,1,1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,1,1,1,-1],
    [1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1],
    [1,1,1,1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1],
    [-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1],
    [1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,1,1],
    [-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1],
    [1,1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1],
    [-1,-1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1],
    [1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1],
    [1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1],
    [-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1],
    [1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1],
    [-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1,-1,1,1],
    [1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1],
    [-1,-1,1,1,1,-1,1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1],
    [1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1],
    [1,1,1,1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,-1,1,1,1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1],
    [-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1],
    [1,1,1,-1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,1,1]]

#declaring neurons
input_array_length = len(input_matrix[0])
letter_a = Neuron()
letter_b = Neuron()
letter_c = Neuron()
letter_d = Neuron()
letter_e = Neuron()
letter_j = Neuron()
letter_k = Neuron()

#setting neurons. Values are changed according to previous trials.
letter_a.set_neuron("A", 1000, 0.01, 3, 0, input_array_length)
letter_b.set_neuron("B", 1000, 0.01, 3, 0, input_array_length)
letter_c.set_neuron("C", 1000, 0.01, 3, 0, input_array_length)
letter_d.set_neuron("D", 1000, 0.01, 3, 0, input_array_length)
letter_e.set_neuron("E", 1000, 0.01, 3, 0, input_array_length)
letter_j.set_neuron("J", 1000, 0.01, 3, 0, input_array_length)
letter_k.set_neuron("K", 1000, 0.01, 3, 0, input_array_length)

#teaching. In every 7 iteration same letter but in a different font comes.
for i in range(len(input_matrix)):
    if i % 7 == 0:
        letter_a.teach_neuron(input_matrix[i], 1) #giving the desired letter, waiting a result which is true
        letter_b.teach_neuron(input_matrix[i], -1) #giving not wanted letters, waiting for false output
        letter_c.teach_neuron(input_matrix[i], -1)
        letter_d.teach_neuron(input_matrix[i], -1)
        letter_e.teach_neuron(input_matrix[i], -1)
        letter_j.teach_neuron(input_matrix[i], -1)
        letter_k.teach_neuron(input_matrix[i], -1)
    elif i % 7 == 1: #same as above
        letter_b.teach_neuron(input_matrix[i], 1)
        letter_a.teach_neuron(input_matrix[i], -1)
        letter_c.teach_neuron(input_matrix[i], -1)
        letter_d.teach_neuron(input_matrix[i], -1)
        letter_e.teach_neuron(input_matrix[i], -1)
        letter_j.teach_neuron(input_matrix[i], -1)
        letter_k.teach_neuron(input_matrix[i], -1)
    elif i % 7 == 2:
        letter_c.teach_neuron(input_matrix[i], 1)
        letter_a.teach_neuron(input_matrix[i], -1)
        letter_b.teach_neuron(input_matrix[i], -1)
        letter_d.teach_neuron(input_matrix[i], -1)
        letter_e.teach_neuron(input_matrix[i], -1)
        letter_j.teach_neuron(input_matrix[i], -1)
        letter_k.teach_neuron(input_matrix[i], -1)
    elif i % 7 == 3:
        letter_d.teach_neuron(input_matrix[i], 1)
        letter_a.teach_neuron(input_matrix[i], -1)
        letter_b.teach_neuron(input_matrix[i], -1)
        letter_c.teach_neuron(input_matrix[i], -1)
        letter_e.teach_neuron(input_matrix[i], -1)
        letter_j.teach_neuron(input_matrix[i], -1)
        letter_k.teach_neuron(input_matrix[i], -1)
    elif i % 7 == 4:
        letter_e.teach_neuron(input_matrix[i], 1)
        letter_a.teach_neuron(input_matrix[i], -1)
        letter_b.teach_neuron(input_matrix[i], -1)
        letter_c.teach_neuron(input_matrix[i], -1)
        letter_d.teach_neuron(input_matrix[i], -1)
        letter_j.teach_neuron(input_matrix[i], -1)
        letter_k.teach_neuron(input_matrix[i], -1)
    elif i % 7 == 5:
        letter_j.teach_neuron(input_matrix[i], 1)
        letter_a.teach_neuron(input_matrix[i], -1)
        letter_b.teach_neuron(input_matrix[i], -1)
        letter_c.teach_neuron(input_matrix[i], -1)
        letter_d.teach_neuron(input_matrix[i], -1)
        letter_e.teach_neuron(input_matrix[i], -1)
        letter_k.teach_neuron(input_matrix[i], -1)
    else:
        letter_k.teach_neuron(input_matrix[i], 1)
        letter_a.teach_neuron(input_matrix[i], -1)
        letter_b.teach_neuron(input_matrix[i], -1)
        letter_c.teach_neuron(input_matrix[i], -1)
        letter_d.teach_neuron(input_matrix[i], -1)
        letter_e.teach_neuron(input_matrix[i], -1)
        letter_j.teach_neuron(input_matrix[i], -1)

#test matrix is some of the instances above. They are a little bit noisy. 8 to 10 digits of 63 digits are changed.
test_matrix = [
    [-1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1],
    [1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1],
    [-1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],
    [1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1],
    [1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1],
    [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1],
    [1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1]
]
#test matrix's desired output.
test_matrix_real = ["A", "B", "C", "D", "E", "J", "K"]
#a tuple array to keep results.
#it will be our guesses of each letter
results = [('X', 0), ('X', 0), ('X', 0), ('X', 0), ('X', 0), ('X', 0), ('X', 0)]
for i in range(0, 7):
    test_array = test_matrix[i]
    #Max accuracy holds the result of most score.
    maxAccuracy = letter_a.test_neuron(test_array)
    #result is to comapre new result to max accuracy.
    result = letter_b.test_neuron(test_array)
    if result[1] > maxAccuracy[1]:
        maxAccuracy = result
    result = letter_c.test_neuron(test_array)
    if result[1] > maxAccuracy[1]:
        maxAccuracy = result
    result = letter_d.test_neuron(test_array)
    if result[1] > maxAccuracy[1]:
        maxAccuracy = result
    result = letter_e.test_neuron(test_array)
    if result[1] > maxAccuracy[1]:
        maxAccuracy = result
    result = letter_j.test_neuron(test_array)
    if result[1] > maxAccuracy[1]:
        maxAccuracy = result
    result = letter_k.test_neuron(test_array)
    if result[1] > maxAccuracy[1]:
        maxAccuracy = result
    #the max accurate result is written to results array.
    results[i] = maxAccuracy

#printing guesses.
print("--------------------------------------------")
print("Actual Letter | Guessed Letter  | Score")
print("--------------------------------------------")

for i in range(0,7):
    print("\t\t"+test_matrix_real[i]+"\t  |\t\t   "+str(results[i][0]) + "\t\t|\t" + str(results[i][1]))

