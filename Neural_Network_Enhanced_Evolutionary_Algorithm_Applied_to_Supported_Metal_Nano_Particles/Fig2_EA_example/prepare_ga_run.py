#!/usr/bin/env python

from ase.ga.data import PrepareDB
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import closest_distances_generator
from ase.ga.utilities import get_all_atom_types
from ase.io import read, write
import numpy as np


db_file = 'gadb.db'

n = 20
atom_numbers =  3 * [78] + 3 * [47] 
slab = read('slab.traj')

v1 = np.array([5., 0., 0.])
v2 = np.array([0., 5., 0.])
v3 = np.array([0., 0., 6.])
p0 = np.array([1.5, 1.5, 8.5])



# define the closest distance two atoms of a given species can be to each other
unique_atom_types = get_all_atom_types(slab, atom_numbers)
cd = closest_distances_generator(atom_numbers=unique_atom_types,
                                     ratio_of_covalent_radii=0.7)
# create the starting population
sg = StartGenerator(slab = slab,
                    atom_numbers = atom_numbers, 
                    closest_allowed_distances = cd,
                    box_to_place_in = [p0, [v1, v2, v3]])

starting_population = []
for i in range(n):
    print 'looking for candidate %2d'%(i)
    t = sg.get_new_candidate()
    starting_population.append(t)

# create the database to store information in
d = PrepareDB(db_file_name=db_file,
              simulation_cell=slab,
              stoichiometry=atom_numbers)


for a in starting_population:
    d.add_unrelaxed_candidate(a)

write('start_pop.traj',starting_population)
