# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 16:23:21 2019

@author: Vovan-i-Venera
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 16:05:53 2019

@author: Vovan-i-Venera
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 17:31:24 2019

@author: Vovan-i-Venera
"""
import random
from numpified_matrix_multiplications import NN_deeper_numpified_DS_solver
from NN_dataset_solver import NN_dataset_solver
from sklearn.datasets import load_iris
iris = load_iris()
iris_x = iris["data"]
iris_y = iris["target"]

random.seed(5)
NN_struct_and_data = NN_dataset_solver(iris_x,iris_y)
NN_struct_and_data.evolve_NNs(n_cycles = 1000,n_children = 15,
                              n_vectors = 100, loss_func = "hybrid",act_func = "relu",
                              progress_view = False, n_neurons = [4,3])

print(NN_struct_and_data.NNs_evaluated())
#print(NN_struct_and_data.NNs())
random.seed(5)
NN_struct_and_data = NN_numpified_DS_solver(digits_x,digits_y)
NN_struct_and_data.evolve_NNs(n_cycles = 1000,n_children = 15,
                              n_vectors = 100, loss_func = "hybrid",act_func = "relu",
                              progress_view = False, n_neurons = [4,3])

print(NN_struct_and_data.NNs_evaluated())

random.seed(5)
NN_struct_and_data = NN_deeper_numpified_DS_solver(digits_x,digits_y)
NN_struct_and_data.evolve_NNs(n_cycles = 1000,n_children = 15,
                              n_vectors = 100, loss_func = "hybrid",act_func = "relu",
                              progress_view = False, n_neurons = [4,3])

print(NN_struct_and_data.NNs_evaluated())
#print(NN_struct_and_data.NNs())