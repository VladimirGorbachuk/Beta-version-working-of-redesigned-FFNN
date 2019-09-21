# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 22:05:56 2019

@author: Vovan-i-Venera
"""
#from NN_initialize import Neural_Network
#from NN_calculations import Neural_answer, NN_performace_estimation
from NN_evolution import Genetic_cross_breeding


import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #OneHotEncoder пока не юзаем
 
   
class NN_dataset_solver(Genetic_cross_breeding):
    """
    этот класс является последним в цепи наследования:
    Neural_Network > Neural_Read_n_Write(Neural_Network) -> 
    NN_performace_estimation(Neural_answer) ->
    Genetic_cross_breeding(NN_performace_estimation)
    Он выполняет функцию интерфейса, обрабатывает набор данных и используя 
    методы родительских классов собирает произвольные нейросети, скрещивает, оценивает
    их, отбирая лучшие.
    Предполагается что он будет использован в ходе следующих шагов:
    первый шаг - инициализация класса на датасете
    (альтернативный вариант предполагает вытаскивание кваргов из готового объекта нейросети)
    далее используется метод evolve
    нужен дополнительный метод, который бы позволял выдавать "ответы" для заданной нейросети
    или для нескольких нейросетей
    
    Пример использования:
    from NN_dataset_solver import NN_dataset_solver
    from sklearn.datasets import load_iris
    iris = load_iris()
    iris_x = iris["data"]
    iris_y = iris["target"]    
    NN_struct_and_data = NN_dataset_solver(iris_x,iris_y,test_size = 0.3)
    NN_struct_and_data.evolve_NNs(n_cycles = 50,n_children = 150,
                                  n_vectors = 100, loss_func = "hybrid",act_func = "relu"
                                  progress_view = True, n_neurons = [3,3])

    """
    def __init__(self,dataset_x, dataset_y, test_size = 0.2):
        """
        при создании класса нужно ввести наборы данных из датасета, размер тестового набора
        в долях от единицы
        """
        self._x_train, self._x_test, self._y_train, self._y_test = self._dataset_preprocess(dataset_x,dataset_y, test_size)
        
    def _dataset_preprocess(self,dataset_x,dataset_y, test_size):
        """
        в этом встроенном методе мы принимаем из датасета
        наборы векторов и лейблов, чтобы 
        1) нормализовать его
        2) разделить на тренировочный и тестовый набор (используя test_size)
        """
        dataset_x_scaled = StandardScaler().fit_transform(dataset_x)
        x_train,x_test, y_train,y_test = train_test_split(dataset_x_scaled,dataset_y,test_size = test_size)
        return x_train,x_test, y_train,y_test
    
    def evolve_NNs(self, n_cycles = 15000,  n_vectors = 80,
                   n_initial_NNs = 50, n_children = 50, n_mutants =5,
                   mutagenity = 0.1,
                   selection_method = "weighted_selection", 
                   breed_method = "breed_random_binary",
                   loss_func = "hybrid", act_func = "relu", n_neurons = [3,3],
                   with_bias = True,
                   progress_view = True):
        """
        Главный публичный метод, после инициализации с датасетом, запускаем этот метод
        который будет генерировать нейросети, отбирать лучшие используя заданные гиперпараметры
        функции потерь 
        пример вызова данного публичного метода:
        NN_struct_and_data.evolve_NNs(n_cycles = 50,n_children = 150,
                                  n_vectors = 100, loss_func = "hybrid",act_func = "relu"
                                  progress_view = True, n_neurons = [3,3])
        значения отсюда собственно и нужно передавать в super().__init__()
        родительскому классу Genetic_cross_breeding(NN_performace_estimation) нужны метапараметры для эволюционного
        алгоритма и количество векторов по которым будет осуществляться оценка 
        (функцией потерь):
        n_cycles = 50,n_children = 150 идут туда
        более высокому родительскому классу NN_performace_estimation(Neural_answer)
        нужен тип функции потерь 
        следующему по иерархии 
        Neural_answer(Neural_Network) нужна функция активации
        Самому главному же классу Neural_Network нужны параметры
        n_neurons = None, n_in = None, n_out = None, with_bias =True
        Реально из всего этого необходимо задать n_neurons!!!
        """
        n_in = len(self._x_train[0])
        n_out = len(set(self._y_train))
        super().__init__(n_cycles = n_cycles, n_vectors = n_vectors,
                         n_initial_NNs = n_initial_NNs, n_children = n_children,
                         n_mutants = n_mutants, selection_method = selection_method, 
                         mutagenity = mutagenity,
                         breed_method = breed_method,
                         loss_func = loss_func, act_func = act_func,
                         n_neurons = n_neurons, n_in = n_in, n_out = n_out, with_bias = with_bias
                         )
        self.progress_view = progress_view
        self._new_NNs = super().generate()
        self._evaluated_by_test_set_new_NNs = []
        for chosen_NN in self._new_NNs:
            self._evaluated_by_test_set_new_NNs.append(super().guess_rate_test(NN=chosen_NN))
        if self.progress_view:
            print(self._evaluated_by_test_set_new_NNs)
        return
    def NNs_evaluated(self):
        """
        этот публичный метод выдаёт нейросети и их оценки
        """
        return self._evaluated_by_test_set_new_NNs
    def NNs(self):
        """
        этот публичный метод выдаёт последнее поколение нейросетей
        """
        return self._new_NNs