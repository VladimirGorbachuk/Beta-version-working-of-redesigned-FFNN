# -*- coding: utf-8 -*-+
"""
Created on Sat Sep 14 15:48:37 2019

@author: Vovan-i-Venera
"""
from NN_calculations import NN_performace_estimation, Neural_answer
import random
from functools import partial

class Genetic_cross_breeding(NN_performace_estimation):
    """
    Этот класс отвечает за скрещивание нейросетей, которые генерируются 
    самым главным родительским классом Neural_Network, а оцениваются 
    классом NN_performace_estimation(Neural_answer)
    """

    
    def __init__(self, n_cycles = None, n_vectors = None,
                         n_initial_NNs = None, n_children = None,
                         n_mutants = None,selection_method = None,
                         mutagenity = None,
                         breed_method = None,
                         loss_func = None, act_func = None,
                         n_neurons = None, n_in = None, n_out = None, with_bias = None):
        """
        тут отбираются методы скрещивания (в данном коде выбор из 1 варианта)
        и выбора родителей (тоже пока что 1 метод)
        Также можно задать число циклов, количество нейросетей в исходной популяции
        количество детей (не будет превышаться это количество и генерироваться будет
        по-моему в 2 раза больше кандидатов)
        также тут задаётся мутагенность (насколько вероятно при мутации что очередной вес
        в нейронах будет заменён произвольным в интервале -1:1)
        и количество мутированных нейросетей которые будут сгенерированы в каждом цикле
        """
        breed_methods = {"breed_random_binary": self._breed_random_binary}
        selection_methods = {"weighted_selection": self._weighted_selection}
        self._n_cycles = n_cycles
        self._n_initial_NNs = n_initial_NNs
        self._n_children = n_children
        self._selection_method = selection_methods[selection_method]
        self._breed_method = breed_methods[breed_method]
        self._mutagenity = mutagenity
        self._n_mutants = n_mutants
        
        """
        собрали метапараметры, но это пока что недоделанный фрагмент!!!!
        """
        super().__init__(loss_func = loss_func, act_func = act_func, n_vectors = n_vectors,
                         n_neurons = n_neurons, n_in = n_in, n_out = n_out, with_bias = with_bias
                         )
        """
        остальные параметры относящиеся к функции потерь используемой при оценке,
        акт. функции нейросети,
        количеству нейронов в нейросети,
        количеству входных сигналов, выходных... 
        всё передаём выше к родительским классам
        """
        self._children = []
        self._evaluation = []
        """
        c этими двумя списками в основном и будут работать встроенные методы
        """
    def _weighted_selection (self):
        """
        один из методов отбора родителей - пропорционально их оценке
        """
        [parent_a,parent_b] = random.choices (self._children, weights = self._evaluation, k=2)        
        return parent_a, parent_b
    
    def _breed_random_binary (self):
        """
        один из методов скрещивания - произвольный выбор весов из родителя 1 или родителя 2
        """
        self._new_children = []
        for breeds in range(self._n_children//2):
            parent_a,parent_b = self._selection_method() 
            child_1 = [random.choice (weights) for weights in zip (parent_a,parent_b)]
            child_2 = [random.choice (weights) for weights in zip (parent_a,parent_b)]
            self._new_children.append (child_1)
            self._new_children.append (child_2)
        return
    
    def _mutate (self):
        for mutations in range(self._n_mutants):
            mutant = random.choice (self._new_children)
            mutated = []
            for weight in mutant:
                [weight_2] = random.choices ([weight, random.random ()], weights = [1,self._mutagenity])
                mutated.append (weight_2)
            self._new_children.append (mutated)
    
    def _tournament_selection (self):
        """
        метод отбора - новую нейросеть сравниваем с любой из набора на каких-нибудь векторах
        если новая нейросеть выигрывает, то она занимает место проигравшей
        """
        for new_child in self._new_children:
            contestant_number = random.choice(range(len(self._children)))
            contestant = self._children[contestant_number]
            """
            опять замена super на self... странно это
            """
            new_child_estimation = 1/self._loss_function(NN=new_child) #АЛЯРМ
            if new_child_estimation > 1/self._loss_function(NN=contestant):
                self._children [contestant_number] = new_child
                self._evaluation [contestant_number] = new_child_estimation
        self._new_children = []
    
    def _generate_random_NNs_initial(self):
        """
        этот метод создаёт изначальный набор произвольных нейросетей, используя
        метод родительского класса из файла NN_initialize.py
        Neural_Network эти методы генерируют произвольные нейросети
        и записывают их в строку
        
        Затем проводится оценка получившихся нейросетей используя классы 
        из файла NN_calculations
        Neural_answer(Neural_Network) -> NN_performace_estimation(Neural_answer)
        """

        for _number_of_initial_NNs in range(self._n_initial_NNs):
            super().build_random_NN ()
            self._children.append(super().write())
        for random_initial_NN in self._children:
            estimation = super()._loss_function(NN=random_initial_NN)
            self._evaluation.append(estimation)
        assert(len(self._evaluation) == len(self._children))
        return
    
    def generate (self):
        self._generate_random_NNs_initial()
        for _evolution_cycle in range(self._n_cycles):
            self._breed_method()
            self._mutate()
            self._tournament_selection()
            if self.progress_view:
                print("current average performance:", sum(self._evaluation) / len(self._evaluation))
        return self._children