bounds = [(0.0001, 0.01),  
          (16, 64),       
          (5, 20)]
types = [float, int, int]
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

optimizer = GlobalBestPSO(n_particles=10, dimensions=3, bounds=bounds, options=options)
optimizer.optimize(fitness_function, iters=10)
