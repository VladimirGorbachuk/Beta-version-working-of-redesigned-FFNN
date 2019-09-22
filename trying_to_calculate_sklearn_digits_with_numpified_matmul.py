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
from numpified_matrix_multiplications import NN_deeper_numpified_DS_solver
from sklearn.datasets import load_digits
digits = load_digits()
digits_x = digits.images
digits_y = digits.target
n_samples = len(digits.images)
digits_x = digits_x.reshape((n_samples, -1))

NN_struct_and_data = NN_deeper_numpified_DS_solver(digits_x,digits_y)
NN_struct_and_data.evolve_NNs(n_cycles = 100,n_children = 150,
                              n_vectors = 100, loss_func = "hybrid",act_func = "relu",
                              progress_view = True, n_neurons = [64,32])

print(NN_struct_and_data.NNs_evaluated())
#print(NN_struct_and_data.NNs())