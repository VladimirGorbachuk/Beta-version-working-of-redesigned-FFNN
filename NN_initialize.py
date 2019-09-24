# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 21:57:45 2019

@author: Vovan-i-Veneraa
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 23:04:14 2019

@author: Vovan-i-Venera
"""

import random #один раз импортировали и хватит, тут надо где-то выставлять 
from collections import namedtuple
from functools import partial
Neuron = namedtuple ("Neuron", ["weights"])

class Neural_Network:
    """
    Самый главный в иерархии класс, от которого все последующие классы наследуют
    класс для рандомизированной сборки нейросети (для первых нейросетей он необходим)
    Кроме того, используется для записи нейросетей в одну строку
    и соответственно для чтения нейросетей из строки
    Это необходимо для скрещивания нейросетей при помощи генетических алгоритмов
    в этом классе хранится распределение нейронов по слоям, весов по нейронам
    в подфункциях заложен дополнительный функционал - чтобы собирать нейросеть
    по весам из генератора. Это нужно чтобы  собрать нейросеть из весов записанных в строку
    """
    
    def __init__ (self, n_neurons = None, n_in = None, n_out = None,
                         with_bias = None):
            """
            на вход принимает:
            1) n_neurons - число слоев в нейросети
            и количества нейронов в каждом слое
            в виде списка, где индексы соответствуют слоям
            0 - первый слой, 1 - второй итд
            2) n_out количество возможных ответов (лейблов) -
            соответствует количеству нейронов в последнем слое
            3) n_in - количество входов, поскольку необязательно число 
            нейронов входного слоя будет равно числу входных скаляров
            4) with_bias - принимает значения True/False, определяет
            будет ли биас и соответственно доп.вес в каждом нейроне соответствующий биасу
            !!! контринтуитивный момент - самый базовый класс не содержит
            информации о том, какая в нейронах активационная функция
            """
            self.n_out = n_out
            self.neurons = n_neurons + [n_out]
            self.layers = len (self.neurons)
            self.input_length = n_in
            self.with_bias = with_bias

    def __repr__(self):
       if self.weights:
           repres =""
           for number_of_layer,layer in enumerate (self.weights):
               repres += "layer number " + str (number_of_layer) + "\n"
               repres += str (layer)
               repres += "\n"
           return repres
       else:
           return ("Empty_NN")

    def _make_one_neuron (self,layer,weight_generator = None):
        """
        чтобы сделать нейрон нужно знать сколько
        значений в него придет чтобы раскидать
        количество коэффициентов (равно количеству
        входных данных + 1 коэффициент для биаса
        если self.with_bias == True)
       
        """
        if layer >0:
            n_inputs = self.neurons [layer-1]
        else:
            n_inputs = self.input_length
        if self.with_bias:
            n_inputs += 1 #если без биасов то не будем делать лишних весов
        if not weight_generator:
            neuron = Neuron (weights = [random.uniform (-1,1) for signals_and_bias
                     in range (n_inputs)])
        else:
            neuron = Neuron (weights = [next (weight_generator) for signals_and_bias
                     in range (n_inputs)])

        return neuron

    def _make_one_layer (self,layer,weight_generator = None):
        """
        теперь будем использовать встроенную функцию
        для собирания отдельного нейрона для
        собирания отдельного слоя нейронной сети
        """
        neurons_in_layer = [self._make_one_neuron(layer,weight_generator) for neuron in range(self.neurons [layer])]
        return neurons_in_layer
           
    def _build_random_or_defined_NN (self,weight_generator = None):
        """
        функция для послойной сборки weight_generator = значения
        весов записанных в строку, подаваемые на вход в виде генератора
        """
        all_neurons_in_NN = [self._make_one_layer (layer, weight_generator = weight_generator ) for layer in range (self.layers)]
        self.weights = all_neurons_in_NN
        return
    
    def build_random_NN (self):
        """
        метод для сборки произвольной сети (вынесен отдельно для удобства)
        """
        self._build_random_or_defined_NN (weight_generator = None)
        return
          
    def write(self):
        """
        генерирует строку из распределения весов по нейронам, а нейронов- по слоям
        получающаяся при этом информация - "нечитаема" без информации,
        хранящейся в данном классе
        """
        one_liner_NN = []
        for layer in self.weights:
            for neuron in layer:
                for weight in neuron.weights:
                        one_liner_NN.append(weight)
        return one_liner_NN
    
    def read(self, one_liner):
        weights_from_one_liner = (weight for weight in one_liner)
        """
        этот метод в будущем будет расширен, но сейчас - он собирает нейросеть из
        записанных в строку значений весов с учётом количества весов у нейронов
        и количества нейронов в каждом слое
        """
        self._build_random_or_defined_NN  (weight_generator=weights_from_one_liner)
        return

