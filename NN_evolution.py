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
        breed_methods = {"breed_random_binary": self._breed_random_binary,
                         "breed_random_K_nary": self._breed_random_K_nary,
                         "breed_hyperangulate":self._breed_hyperangulate,
                         "breed_binary_cross_sect": self._breed_binary_cross_sect,
                         "breed_binary_summ":self._breed_binary_summ}
        selection_methods = {"weighted_selection": self._weighted_selection,
                             "deviant_selection":self._deviant_selection}
        self._n_cycles = n_cycles
        self._n_initial_NNs = n_initial_NNs
        self._n_children = n_children
        self._selection_method = selection_methods[selection_method]
        self._breed_method = breed_methods[breed_method]
        self._mutagenity = mutagenity
        self._n_mutants = n_mutants
        

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
    def _weighted_selection (self, k_parents):
        """
        один из методов отбора родителей - пропорционально их оценке, пожалуй самый банальный
        """
        parents = random.choices (self._children, weights = self._evaluation, k=k_parents)        
        return parents
    
    def _deviant_selection (self, k_parents):
        """
        ВМЕСТО ДОКСТРИНГА тут будет рефлексия:
        поговаривают, что в эволюционных алгоритмах главная проблема - разнообразие решений
        а точнее: главная фишка - в разнообразии решений, и эту фишку мягко говоря легко 
        потерять. Конечно можно ждать когда мутация сгенерирует подходящего мутанта... 
        но прямо скажем, это не самый здравый подход.
        Хотя я пока слабо понимаю, как сделать имитацию отжига (это обязательно нужно в будущем, т.к.
        это единственный скоростной вариант из самых попсовых)
        ничего не мешает мне сотворить свой вариант отбора родителей. Вес будет пропорционален
        среднеквадратичному расстоянию
        """
        parents = []
        [parent_a] = random.choices(self._children, weights = self._evaluation, k = 1)
        parents.append(parent_a)
        deviant_weights = []
        current_point_weight_average = parent_a
        for n_parent in range(1,k_parents):
            deviant_weights = []
            for mb_parent in self._children:
                deviant_weight = sum ([(w1-w2)**2 for w1, w2 in zip (current_point_weight_average,mb_parent)])
                deviant_weights.append(deviant_weight)
            [parent_additional] = random.choices(self._children, weights = deviant_weights, k = 1)
            current_point_weight_average = [w_avg*(n_parent-1)/n_parent + w_add/n_parent 
                                            for w_avg,w_add in zip (current_point_weight_average,parent_additional)]
            parents.append(parent_additional)
        return parents
    
    def _breed_random_binary (self):
        """
        самый распространённый, самый банальный метод скрещивания - произвольный выбор весов из родителя 1 или родителя 2
        """
        self._new_children = []
        for breeds in range(self._n_children//2):
            parent_a,parent_b = self._selection_method(2) 
            child_1 = [random.choice (weights) for weights in zip (parent_a,parent_b)]
            child_2 = [random.choice (weights) for weights in zip (parent_a,parent_b)]
            self._new_children.append (child_1)
            self._new_children.append (child_2)
        return
    
    def _breed_binary_cross_sect (self):
        self._new_children = []
        for breeds in range(self._n_children//2):
            parent_a,parent_b = self._selection_method(2)
            deliminator = random.choice (range (len (parent_a)))
            child_1 = parent_a [:deliminator]+parent_b [deliminator:]
            child_2 = parent_b [:deliminator]+parent_a [deliminator:]
            self._new_children.append (child_1)
            self._new_children.append (child_2)
        return
    
    def _breed_binary_summ (self):
        self._new_children = []
        for breeds in range(self._n_children//2):
            parent_a,parent_b = self._selection_method(2)
            multiplier = random.random()
            child_1 = [w1*multiplier+w2*(1-multiplier) for w1,w2 in zip(parent_a,parent_b)]
            child_2 = [w2*multiplier+w1*(1-multiplier) for w1,w2 in zip(parent_a,parent_b)]
            self._new_children.append (child_1)
            self._new_children.append (child_2)
        return
    
    def _breed_random_K_nary (self):
        """
        один из методов скрещивания - произвольный выбор весов из k родителей, где k рандомное
        целое число от 2 до 10, посмотрим - может так быстрее сходится?
        """
        self._new_children = []
        for breeds in range(self._n_children//5+1):
            k_parents = random.randint (2,10)
            parents = self._selection_method(k_parents) 
            for child in range(k_parents):
                new_child = [random.choice (weights) for weights in zip(*parents)]
                self._new_children.append (new_child)
        return
    
    def _breed_hyperangulate (self):
        """
        по идее этот метод должен очень быстро сходится в какой-нибудь локальный минимум, т.к.
        по сути каждый раз мы ищем среднее между несколькими нейросетями. Эдакий аналог триангуляции
        """
        self._new_children = []
        
        for breeds in range(self._n_children//2):
            k_parents = random.randint (2,10)
            parents = self._selection_method(k_parents) 
            child = []
            for weights in zip (*parents):
                child. append (sum (weights)/k_parents)
            self._new_children.append (child)
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
            estimation = 1/super()._loss_function(NN=random_initial_NN)
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