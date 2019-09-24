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
from NN_override_np_tf import NN_numpified_DS_solver, NN_deeper_numpified_DS_solver, Deeper_Tensorflowed
from NN_dataset_solver import NN_dataset_solver
from sklearn.datasets import load_iris
iris = load_iris()
iris_x = iris["data"]
iris_y = iris["target"]


NN_struct_and_data = NN_dataset_solver(iris_x,iris_y)
NN_struct_and_data.evolve_NNs(n_cycles = 1000,n_children = 15,selection_method = "deviant_selection", 
                              n_vectors = 10, loss_func = "hybrid",act_func = "relu",
                              progress_view = False, n_neurons = [4])

print(NN_struct_and_data.NNs_evaluated())
#print(NN_struct_and_data.NNs())

NN_struct_and_data = NN_numpified_DS_solver(iris_x,iris_y)
NN_struct_and_data.evolve_NNs(n_cycles = 1000,n_children = 15,selection_method = "deviant_selection",
                              n_vectors = 10, loss_func = "hybrid",act_func = "relu",
                              progress_view = False, n_neurons = [4])

print(NN_struct_and_data.NNs_evaluated())


NN_struct_and_data = NN_deeper_numpified_DS_solver(iris_x,iris_y)
NN_struct_and_data.evolve_NNs(n_cycles = 1000,n_children = 15,selection_method = "deviant_selection",
                              n_vectors = 10, loss_func = "hybrid",act_func = "relu",
                              progress_view = False, n_neurons = [4])

print(NN_struct_and_data.NNs_evaluated())
#print(NN_struct_and_data.NNs())

NN_struct_and_data = Deeper_Tensorflowed(iris_x,iris_y)
NN_struct_and_data.evolve_NNs(n_cycles = 1000,n_children = 15,selection_method = "deviant_selection",
                              n_vectors = 10, loss_func = "hybrid",act_func = "relu",
                              progress_view = False, n_neurons = [4])

print(NN_struct_and_data.NNs_evaluated())

