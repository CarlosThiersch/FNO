# -*- coding: utf-8 -*-
"""GeradorSoluçõesDeOnda.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yT0RCgl4NkknhlwDpSj6YFtAtnV8uAv8
"""

import numpy as np
import h5py 

import random

import devito
from devito import TimeFunction, Eq, solve, Operator
from examples.seismic import Model, TimeAxis, RickerSource, Receiver

from timeit import default_timer
hf1 = h5py.File('./data/20000VM-64.hdf5','r') 
field = np.array(hf1.get('data'))
hf1.close()

hf = h5py.File('./data/wave6464Solution12000.hdf5','w')

n = 64
m = 40
shape = (n,n)
spacing = (m,m)
origin = (0.,0,)
f0 = 0.005  # Source peak frequency is 5Hz (0.005 kHz)
a = []
for i in range(n):
  for j in range(n):
   a.append(i*m)

t0 = 0.  # Simulation starts a t=0
tn = 4000.  # Simulation last 4 second (4000 ms)
dt = 4  # Time step from model grid spacing

time_range = TimeAxis(start=t0, stop=tn, step=dt)
time = []
for i in range(0,12000):
  model = Model(vp=np.transpose(field[i]/1000), origin=origin, shape=shape, spacing=spacing,
              space_order=2,nbl=120, bcs="damp")
  src = RickerSource(name='src', grid=model.grid, f0=f0,
                   npoint=1, time_range=time_range)
  sourcex = random.randint(0,63)
  sourcey = random.randint(0,63)
  src.coordinates.data[0, :] = sourcey*40
  src.coordinates.data[0, -1] = sourcex

  # o seguinte trecho posiciona um receptor em cada ponto do grid:
  rec = Receiver(name='rec', grid=model.grid, npoint=n**2, time_range=time_range)

  for k in range(n**2):
   rec.coordinates.data[k,0] = a[k]
  for j in range(n):
   rec.coordinates.data[n*j:n*(j+1),1] = np.linspace(0, model.domain_size[0], num=n)

  u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=2)

# We can now write the PDE
  pde = model.m * u.dt2 - u.laplace + model.damp * u.dt

  stencil = Eq(u.forward, solve(pde, u.forward))

# Finally we define the source injection and receiver read function to generate the corresponding code
  src_term = src.inject(field=u.forward, expr=src * dt**2 / model.m)

# Create interpolation expression for receivers
  rec_term = rec.interpolate(expr=u.forward)

  op = Operator([stencil] + src_term + rec_term, subs=model.spacing_map)

#uso do operador para resolução
  print(i)
  # We can now write the PDE
  inicio = default_timer()
  op(time=time_range.num-1, dt=dt)
  time.append(default_timer() - inicio)
  name = 'data_set' + str(i)
  source_coor = np.zeros((64,64))
  source_coor[sourcex,sourcey] = 1

  hf.create_dataset(name + 'source',data = np.concatenate((source_coor.reshape(64,64,1),(field[i]/1000).reshape(64,64,1)),axis = -1))
  hf.create_dataset(name,data = np.transpose(np.array(rec.data[:-1]).reshape(1000,64,64),(0,2,1)))
hf.close()
time = np.array(time)
print(np.sum(time),np.average(time),np.std(time))