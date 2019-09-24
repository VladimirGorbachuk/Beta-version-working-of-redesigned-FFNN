# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 00:42:22 2019

@author: Vovan-i-Venera
"""


from NN_dataset_solver import NN_dataset_solver
from sklearn.datasets import load_digits
digits = load_digits()
digits_x = digits.images
digits_y = digits.target
n_samples = len(digits.images)
digits_x = digits_x.reshape((n_samples, -1))

NN_struct_and_data = NN_dataset_solver(digits_x,digits_y)
NN_struct_and_data.evolve_NNs(n_cycles = 1000,n_children = 50,
                              breed_method = "breed_random_K_nary", selection_method = "deviant_selection",
                              n_vectors = 20, loss_func = "hybrid",act_func = "relu",
                              progress_view = True, n_neurons = [64,32])

print(NN_struct_and_data.NNs_evaluated())
#print(NN_struct_and_data.NNs())
