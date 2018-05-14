from random import random
from ase.io import write
from ase.optimize import BFGS
from ase.calculators.emt import EMT

from ase.ga.data import DataConnection
from ase.ga.population import Population
from ase.ga.standard_comparators import InteratomicDistanceComparator
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.utilities import closest_distances_generator, get_all_atom_types
from ase.ga.standardmutations import PermutationMutation

# Initialize the different components of the GA
da = DataConnection('gadb.db')

# Relax all unrelaxed structures (e.g. the starting population)
k = 0
while da.get_number_of_unrelaxed_candidates() > 0:
    a = da.get_an_unrelaxed_candidate()
    a.set_calculator(EMT())
    dyn = BFGS(a, logfile=None, trajectory='start_pop' + str(k) + '.traj')
    dyn.run(fmax=0.025)
    a.info['key_value_pairs']['raw_score'] = -a.get_potential_energy()
    da.add_relaxed_step(a)
    k += 1
    print 'start', k
# Create the population
atom_numbers_to_optimize = da.get_atom_numbers_to_optimize()
n_to_optimize = len(atom_numbers_to_optimize)
slab = da.get_slab()
all_atom_types = get_all_atom_types(slab, atom_numbers_to_optimize)
blmin = closest_distances_generator(all_atom_types, ratio_of_covalent_radii=0.7)
comp = InteratomicDistanceComparator(n_top=n_to_optimize)

population = Population(data_connection=da, population_size=20, comparator=comp)
pairing = CutAndSplicePairing(slab, n_to_optimize, blmin)
mutations = PermutationMutation(n_to_optimize)

# Test 20 new candidates
for i in range(400):
    a1, a2 = population.get_two_candidates()

    # Check if we want to do a mutation or pair candidates
    if random() < 0.3:
        a3_mut, desc = mutations.get_new_individual([a1])
        if a3_mut is not None:
            da.add_unrelaxed_step(a3_mut, desc)
            a3 = a3_mut
        else:
            continue
    else:
        a3, desc = pairing.get_new_individual([a1, a2])
        if a3 is None:
            continue
        da.add_unrelaxed_candidate(a3, description=desc)
        
    # Relax the new candidate
    a3.set_calculator(EMT())
    dyn = BFGS(a3, logfile=None)
    dyn.run(fmax=0.025)
    a3.info['key_value_pairs']['raw_score'] = -a3.get_potential_energy()
    da.add_relaxed_step(a3)
    population.update()
    print i, round(a3.get_potential_energy(),3)

write('all_candidates.traj', da.get_all_relaxed_candidates())
