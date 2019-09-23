# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 13:07:14 2019

@author: Vovan-i-Venera
"""
from NN_initialize import Neural_Network
import random
from math import exp, log

class Neural_answer(Neural_Network):
    """
    этот класс отвечает за выдачу ответа, соответствующего входным данным и отдельно
    взятой нейросети, содержит словарь функций активации
    """

    """
    берём нейросеть, при помощи встроенной функции calc_output расчитываем для неё результат
    """
    def __init__(self, act_func = None,
                         n_neurons = None, n_in = None, n_out = None,
                         with_bias = None):
        """
        в этом классе нужно только одно значение - тип функции активации,
        которая будет для всех нейронов одинаковой (это конечно можно расширить... 
        но пока что не нужно)
        """
        act_func_types = {"relu": self._relu, 
                            "elu": self._elu,
                            "leaky_relu": self._leaky_relu,
                            "sigmoid": self._sigmoid,
                            "binary": self._binary,
                            "tanh": self._tanh,
                            "softplus": self._softplus
                            }
        self._activation_function = act_func_types[act_func]
        super().__init__(n_neurons = n_neurons, n_in = n_in, n_out = n_out,
                         with_bias = with_bias)
    """
    далее идёт стандартный набор функций активации
    """
    
    def _relu (self, output):
        output = [x if x>0 else 0 for x in output ]
        return output
    
    def _elu(self, output):
        output = [x if x> 0 else 0.01*(exp(x)-1) for x in output]
        return output
    
    def _leaky_relu (self, output):
        output = [x if x>0 else 0.01*x for x in output]
        return output
    
    def _sigmoid (self, output):
        output = [1/(1+exp(-x)) for x in output ]
        return output
        
    def _binary (self,output):
        output = [1 if x>0 else 0 for x in output ]
        return output
    
    def _softplus (self, output):
        output = [log(1+exp(x)) for x in output]
        return output

    def _tanh (self, output):
        output = [exp(2*x - 1)/exp(2*x + 1) for x in output]
        return output
    
    
    
    
    def _matrix_mult_act_func (self,input,layer_number = None):
        """
        самый простой вариант перемножения матриц с биасом 
        (позже нужно будет сюда пилить альтернативу на нумпае)
        """
        if self.with_bias and layer_number < self.layers:
            input = list (input)
            input.insert (0,1)
        output = []
        for neuron in self.weights[layer_number]:
            product = 0
            for number_of_scalar,scalar in enumerate (input):
                product += neuron.weights [number_of_scalar]*scalar
            output.append (product)
        return self._activation_function(output)
    


    def calc_output (self,input):
        """
        рассчитываем числа, которые выдаёт нейросеть для данного input (одномерного вектора)
        если в виде одной строки нейросеть, то используем наследованную функцию self.read()
        """

        for layer_number in range (self.layers):
            input = self._matrix_mult_act_func (input,layer_number = layer_number)       
        return input
    
    def nn_answer (self, input):
        """
        используем встроенную функцию calc_output, чтобы выяснить какой ответ
        даёт нейросеть - индекс максимального значения - соответствует ответу нейросети
        если в виде одной строки нейросеть, то используем наследованную функцию self.read()
        """

        output = self.calc_output(input)
        answer = output.index(max(output))
        return answer
    
    
class NN_performace_estimation(Neural_answer):
    


    """
    этот класс использует
    методы calc_output и nn_answer 
    родительского класса чтобы оценить
    насколько хорошо работает очередная нейросеть
    выдаёт MAE и долю верно отгаданных ответов.
    """
    def __init__(self, loss_func = None, act_func = None,
                         n_neurons = None, n_in = None, n_out = None,
                         with_bias = None, n_vectors = None):
        """
        при инициализации этому класссу нужен только тип активационной функции
        и количество векторов, по которым будет проводиться оценка
        """

        super().__init__(act_func = act_func,
                         n_neurons = n_neurons, n_in = n_in, n_out = n_out,
                         with_bias = with_bias)
        self._loss_function_types = {"MAE":self.mean_abs_error,
                                 "MSE": self.mean_squared_error,
                                 "hybrid": self.hybrid_MAE_GR}
        self._loss_function_type = self._loss_function_types[loss_func]
        self._n_vectors = n_vectors
        
    def numbers_of_n_vectors_chosen(self):
        """
        этот метод выбирает те векторы, для которых будет проводиться оценка нейросети
        количество векторов равно self._n_vectors
        """
        numbers_of_vectors_chosen = random.choices(range(len(self._x_train)), k = self._n_vectors)
        return numbers_of_vectors_chosen 
    
    def _loss_function(self, NN):
        """
        тут считывается нейросеть, отбираются произвольные вектора вызывается
        """
        super().read(NN)
        numbers_of_vectors_chosen = self.numbers_of_n_vectors_chosen()
        return self._loss_function_type(numbers_of_vectors_chosen)
    
    def _NN_answers_and_real_labels(self, numbers_of_vectors_chosen = None):
        """
        вынес эту часть чтобы сделать более глубокий и удобный оверрайд
        это самое долгое- определить ответы которые даст отдельно взятая нейросеть
        поэтому я хочу их сгруппировать, чтобы их было легче перебить в формат
        умножения массивов нампи (где первый массив - сразу все нужные векторы)
        и уж тем более (за счёт распараллеливания) это нужно в tensorflow
        """
        
        nn_answers = []
        labels = []
        for number in numbers_of_vectors_chosen:
            correct_one_hot_encoded_answer = [0 if n_answer != self._y_train[number] else 1 for n_answer in range(self.n_out) ]
            nn_answer = self.calc_output(self._x_train[number])
            nn_answers.append(nn_answer)
            labels.append(correct_one_hot_encoded_answer)
        return nn_answers, labels
    
    def mean_abs_error (self,  numbers_of_vectors_chosen = None):
        """
        среднее отклонение по модулю разности, используется только для тренировочного набора
        """
        abs_error = 0
        nn_answers, labels = self._NN_answers_and_real_labels(numbers_of_vectors_chosen)
        for answer_label_pair_index in range(len(numbers_of_vectors_chosen)):
            correct_one_hot_encoded_answer = labels[answer_label_pair_index]
            nn_answer = nn_answers[answer_label_pair_index]
            abs_error += sum([abs(val2-val1) for val1, val2 in zip(nn_answer,correct_one_hot_encoded_answer)])/self.n_out
        mean_abs_error = abs_error / self._n_vectors
        return mean_abs_error
    
    def mean_squared_error (self,  numbers_of_vectors_chosen = None):
        """
        среднее отклонение по квадрату разности, используется только для тренировочного набора
        """
        sq_error = 0
        nn_answers, labels = self._NN_answers_and_real_labels(numbers_of_vectors_chosen)
        for answer_label_pair_index in range(len(numbers_of_vectors_chosen)):
            correct_one_hot_encoded_answer = labels[answer_label_pair_index]
            nn_answer = nn_answers[answer_label_pair_index]
            sq_error += sum([(val2-val1)**2 for val1, val2 in zip(nn_answer,correct_one_hot_encoded_answer)])/self.n_out
        mean_sq_error = sq_error / self._n_vectors
        return mean_sq_error
    
    def hybrid_MAE_GR (self,  numbers_of_vectors_chosen = None):
        """
        гибрид средней абсолютной ошибки и доли отгадывания - 
        число обратное доле отгаданных ответов умножается на среднее отклонение
        по модулю
        """
        abs_error = 0
        total_guesses = 0
        correct_guesses = 0
        nn_answers, labels = self._NN_answers_and_real_labels(numbers_of_vectors_chosen)
        for answer_label_pair_index in range(len(numbers_of_vectors_chosen)):
            correct_one_hot_encoded_answer = labels[answer_label_pair_index]
            nn_answer = nn_answers[answer_label_pair_index]
            total_guesses += 1
            abs_error += sum([abs(val2-val1) for val1, val2 in zip(nn_answer,correct_one_hot_encoded_answer)])/self.n_out
            if nn_answer.index(max(nn_answer)) == self._y_train[answer_label_pair_index ]:
                correct_guesses += 1
        mean_abs_error = abs_error / self._n_vectors
        return mean_abs_error*total_guesses/(correct_guesses+1)   
    
    def guess_rate_test(self, NN=None):
        """
        доля отгаданных верных ответов (1 соответствует 100% верно отгаданных ответов)
        этот метод используется только для тестового набора (целиком)
        """
        super().read(NN)
        correct_guesses = 0
        total_guesses = 0
        for number in range(len(self._x_test)):
            if self.nn_answer(self._x_test[number]) == self._y_test[number]:
                correct_guesses += 1
            total_guesses +=1
        return correct_guesses / total_guesses
