# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 17:31:24 2019

@author: Vovan-i-Venera
"""
from NN_dataset_solver import NN_dataset_solver
from sklearn.datasets import load_iris
iris = load_iris()
iris_x = iris["data"]
iris_y = iris["target"]

NN_struct_and_data = NN_dataset_solver(iris_x,iris_y)
NN_struct_and_data.evolve_NNs(n_cycles = 100,n_children = 15,
                              n_vectors = 100, loss_func = "hybrid",act_func = "relu",
                              progress_view = True, n_neurons = [2,3])

print(NN_struct_and_data.NNs_evaluated())
#print(NN_struct_and_data.NNs())
