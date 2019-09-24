# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 12:34:56 2019

@author: Vovan-i-Venera
"""


from NN_dataset_solver import NN_dataset_solver
from sklearn.datasets import load_iris
iris = load_iris()
iris_x = iris["data"]
iris_y = iris["target"]

NN_struct_and_data = NN_dataset_solver(iris_x,iris_y)
NN_struct_and_data.evolve_NNs(n_cycles = 10,n_children = 15,
                              n_vectors = 20, loss_func = "hybrid",act_func = "relu",
                              breed_method = "breed_hyperangulate", selection_method = "deviant_selection",
                              progress_view = True, n_neurons = [2,3])

print(NN_struct_and_data.NNs_evaluated())
#print(NN_struct_and_data.NNs())



NN_struct_and_data = NN_dataset_solver(iris_x,iris_y)
NN_struct_and_data.evolve_NNs(n_cycles = 10,n_children = 15,
                              n_vectors = 20, loss_func = "hybrid",act_func = "relu",
                              breed_method = "breed_random_K_nary", selection_method = "deviant_selection",
                              progress_view = True, n_neurons = [2,3])

print(NN_struct_and_data.NNs_evaluated())

NN_struct_and_data = NN_dataset_solver(iris_x,iris_y)
NN_struct_and_data.evolve_NNs(n_cycles = 10,n_children = 15,
                              n_vectors = 20, loss_func = "hybrid",act_func = "relu",
                              breed_method = "breed_binary_cross_sect", selection_method = "deviant_selection",
                              progress_view = True, n_neurons = [2,3])

print(NN_struct_and_data.NNs_evaluated())

NN_struct_and_data = NN_dataset_solver(iris_x,iris_y)
NN_struct_and_data.evolve_NNs(n_cycles = 10,n_children = 15,
                              n_vectors = 20, loss_func = "hybrid",act_func = "leaky_relu",
                              breed_method = "breed_random_binary", selection_method = "deviant_selection",
                              progress_view = True, n_neurons = [2,3])

print(NN_struct_and_data.NNs_evaluated())

NN_struct_and_data = NN_dataset_solver(iris_x,iris_y)
NN_struct_and_data.evolve_NNs(n_cycles = 10,n_children = 15,
                              n_vectors = 20, loss_func = "hybrid",act_func = "elu",
                              breed_method = "breed_random_K_nary", selection_method = "weighted_selection",
                              progress_view = True, n_neurons = [2,3])

print(NN_struct_and_data.NNs_evaluated())

NN_struct_and_data = NN_dataset_solver(iris_x,iris_y)
NN_struct_and_data.evolve_NNs(n_cycles = 10,n_children = 15,
                              n_vectors = 20, loss_func = "hybrid",act_func = "tanh",
                              breed_method = "breed_random_binary", selection_method = "deviant_selection",
                              progress_view = True, n_neurons = [2,3])

print(NN_struct_and_data.NNs_evaluated())

NN_struct_and_data = NN_dataset_solver(iris_x,iris_y)
NN_struct_and_data.evolve_NNs(n_cycles = 10,n_children = 15,
                              n_vectors = 20, loss_func = "hybrid",act_func = "sigmoid",
                              breed_method = "breed_random_K_nary", selection_method = "weighted_selection",
                              progress_view = True, n_neurons = [2,3])

print(NN_struct_and_data.NNs_evaluated())


NN_struct_and_data = NN_dataset_solver(iris_x,iris_y)
NN_struct_and_data.evolve_NNs(n_cycles = 10,n_children = 15,
                              n_vectors = 20, loss_func = "hybrid",act_func = "binary",
                              breed_method = "breed_random_binary", selection_method = "deviant_selection",
                              progress_view = True, n_neurons = [2,3])

print(NN_struct_and_data.NNs_evaluated())


NN_struct_and_data = NN_dataset_solver(iris_x,iris_y)
NN_struct_and_data.evolve_NNs(n_cycles = 10,n_children = 15,
                              n_vectors = 20, loss_func = "MSE",act_func = "softplus",
                              breed_method = "breed_random_binary", selection_method = "deviant_selection",
                              progress_view = True, n_neurons = [2,3])

print(NN_struct_and_data.NNs_evaluated())


NN_struct_and_data = NN_dataset_solver(iris_x,iris_y)
NN_struct_and_data.evolve_NNs(n_cycles = 10,n_children = 15,
                              n_vectors = 20, loss_func = "MAE",act_func = "relu",
                              breed_method = "breed_random_binary", selection_method = "deviant_selection",
                              progress_view = True, n_neurons = [2,3])

print(NN_struct_and_data.NNs_evaluated())

NN_struct_and_data = NN_dataset_solver(iris_x,iris_y)
NN_struct_and_data.evolve_NNs(n_cycles = 10,n_children = 15,
                              n_vectors = 20, loss_func = "MAE",act_func = "relu",
                              breed_method = "breed_binary_summ", selection_method = "deviant_selection",
                              progress_view = True, n_neurons = [2,3])

print(NN_struct_and_data.NNs_evaluated())

NN_struct_and_data = NN_dataset_solver(iris_x,iris_y)
NN_struct_and_data.evolve_NNs(n_cycles = 10,n_children = 15,
                              n_vectors = 20, loss_func = "MAE",act_func = "relu",
                              breed_method = "breed_binary_summ", selection_method = "weighted_selection",
                              progress_view = True, n_neurons = [2,3])

print(NN_struct_and_data.NNs_evaluated())
