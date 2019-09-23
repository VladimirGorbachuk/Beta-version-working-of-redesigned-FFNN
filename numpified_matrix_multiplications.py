# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 15:19:17 2019

@author: Vovan-i-Venera
"""

from NN_dataset_solver import NN_dataset_solver
import numpy as np
class NN_numpified_DS_solver(NN_dataset_solver):
    """
    наследует от самого крайнего класса, должен
    оверрайднуть именно операцию перемножения матриц (остальное пока никак не трогаем)
    это очень очень очень поверхностный оверрайд!
    """
    def _matrix_mult_act_func (self,input,layer_number = None):
        """
        самый простой вариант перемножения матриц с биасом 
        (позже нужно будет сюда пилить альтернативу на нумпае)
        """
        input = list(input)
        if self.with_bias and layer_number < self.layers:
            input.insert(0,1)
        matrix_of_neurons = []
        for neuron in self.weights[layer_number]:
            matrix_of_neurons.append(*neuron)
        matrix = np.asarray(matrix_of_neurons)
        input = np.asarray(input)
        output = input@matrix.T
        output = list(output)
        return self._activation_function(output)
    
class NN_deeper_numpified_DS_solver(NN_dataset_solver):
    """
    наследует от самого крайнего класса, должен
    оверрайднуть сразу три метода:
        - и перемножение матриц
        - и обращающийся к нему расчёт выходных данных
        - и обволакивающий расчёт ответов для заданных рандомных векторов
    зачем это всё? чтобы заменить перемножением матрицы всех рандомных векторов на 
    матрицу отдельновзятой нейросети!
    """
    def _matrix_mult_act_func (self,input,layer_number = None):
        """
        самый простой вариант перемножения матриц с биасом 
        (позже нужно будет сюда пилить альтернативу на нумпае)
        """
        if self.with_bias and layer_number < self.layers:    
            input = np.c_[np.ones(len(input)),input]
        matrix_of_neurons = []
        for neuron in self.weights[layer_number]:
            matrix_of_neurons.append(*neuron)
        matrix = np.asarray(matrix_of_neurons)
        output = input@matrix.T
        answer = []
        for single_answer in output:
            answer.append(self._activation_function(single_answer))
        return answer
    
    def calc_output (self,numbers_of_vectors_chosen = None, test = False):
        """
        рассчитываем числа, которые выдаёт нейросеть для данного input (одномерного вектора)
        если в виде одной строки нейросеть, то используем наследованную функцию self.read()
        """
        input = []
        if test:
            input = self._x_test
        else:
            for number in numbers_of_vectors_chosen:
                input.append(self._x_train[number])
        for layer_number in range (self.layers):
            input = self._matrix_mult_act_func (input,layer_number = layer_number)       
        return input
    
    def _NN_answers_and_real_labels(self, numbers_of_vectors_chosen = None):
        """
        вынес эту часть чтобы сделать более глубокий и удобный оверрайд
        это самое долгое- определить ответы которые даст отдельно взятая нейросеть
        поэтому я хочу их сгруппировать, чтобы их было легче перебить в формат
        умножения массивов нампи (где первый массив - сразу все нужные векторы)
        и уж тем более (за счёт распараллеливания) это нужно в tensorflow
        """
        
        nn_answers = self.calc_output(numbers_of_vectors_chosen)
        labels = []
        for number in numbers_of_vectors_chosen:
            correct_one_hot_encoded_answer = [0 if n_answer != self._y_train[number] else 1
                                              for n_answer in range(self.n_out)]
            labels.append(correct_one_hot_encoded_answer)
        return nn_answers, labels
    
    def guess_rate_test(self, NN=None):
        """
        класс не изменён кроме обращений к nn_answer заменённыз 
        доля отгаданных верных ответов (1 соответствует 100% верно отгаданных ответов)
        этот метод используется только для тестового набора (целиком)
        """
        super().read(NN)
        correct_guesses = 0
        total_guesses = 0
        nn_answers = self.calc_output(numbers_of_vectors_chosen = None, test = True)
        for number in range(len(self._x_test)):
            if nn_answers[number].index(max(nn_answers[number])) == self._y_test[number]:
                correct_guesses += 1
            total_guesses +=1
        return correct_guesses / total_guesses
    