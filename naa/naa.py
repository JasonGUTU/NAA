# -*- coding: utf-8 -*-
"""
Author : JasonGUTU
Email  : hellojasongt@gmail.com
Python : anaconda3
Date   : 2016/11/18
"""
import random

import numpy as np


class NAA_base(object):

    def __init__(self, dimension, bound, iteration, parameters, verbose=None, acpt_rnge=None):
        """NAA base class, include population initialize and basic evolution algorithm.

        For now, all the dimension default to be float or np.float64 type.

        Args:
             dimension  - type: integer. Dimention of the optimization problem.
             bound      - type: np.ndarray or two-dimensional-list. The boundaries of each dimension.
                          example: [lower_bound_list, upper_bound_list]
             iteration  - type: integer. Maximum iteration times
             parameters - type: list. Control parameters of NAA. it includes
                            [(1) population       - integer. Number of the individuals,
                             (2) shelter_number   - integer. Number of the shelters,
                             (3) shelter_capacity - integer. Capacity of each shelter,
                             (4) Cr_local         - float (0, 1] Local crossover factor,                      
                             (5) Cr_global        - float (0, 1] Global crossover factor,
                             (6) alpha            - float. Movement factor,
                             (7) delta            - float. Scaling factor]
             verbose    - iteration information display flag.                                    TODO: verbose
             acpt_rnge  - Acceptance range, default to be None
        """
        assert isinstance(dimension, int), "The NAA_base init argument `dimension` must be an integer."
        self.dim = dimension  # dimension of problem
        assert len(bound) == 2, "The NAA init require both upper and lower bounds to go."
        assert len(bound[0]) == dimension and len(bound[1]) == dimension, "The dimension of the lower bound and upper bound should be same."
        self.up_bound = np.array(bound[1])  # upper bound
        self.low_bound = np.array(bound[0])  # lower bound
        self.diff_bound = self.up_bound - self.low_bound  # different between two bounds
        assert all(self.diff_bound > 0), "Upper bound must larger than lower bound. Input should be `[lower_bound_list, upper_bound_list]`"
        try:
            self.iter_time = int(iteration)  # Maximum iteration times
        except:
            raise ValueError("Argument `iteration` can not convert into a int.")
        parameters_type = [int, int, int, float, float, float]
        for i in range(6):
            assert isinstance(parameters[i], parameters_type[i]), "The %d th parameter in `parameters` has a wrong type, expect %s." % (i, str(parameters_type[i]))
        self.N_pop = parameters[0]  # Number of the individuals
        self.N_shels = parameters[1]  # Number of the shelters
        self.cap = parameters[2]  # Capacity of each shelter
        assert 0 < parameters[3] < 1.0 and 0 < parameters[4] < 1.0, "The Crossover Factor `parameters[3]` or `parameters[4]` should be in (1, 0)."
        self.cr_lc = parameters[3]  # Local crossover factor
        self.cr_gb = parameters[4]  # Global crossover factor
        self.alpha = parameters[5]  # Movement factor
        self.delta = parameters[6]  # Scaling factor

        # Acceptance range
        if acpt_rnge is not None:
            assert isinstance(acpt_rnge, float), "The Acceptance range `acpt_rnge` must be float."  # TODO: non-float?
            self.acpt_rnge = acpt_rnge
        else:
            self.acpt_rnge = None
        # check is the problem loaded successfully
        self.loaded = False

    def load_problem(self, opt_objective, fitness_func=None):
        """Load optimization problem and fitness function.

        Args:
            opt_objective - type: function.
            fitness_func  - type: function.
        """
        # TODO : test if callable
        assert hasattr(opt_objective, '__call__'), "`opt_objective` must be callable."
        if fitness_func is not None:
            assert hasattr(fitness_func, '__call__'), "`fitness_func` must be callable."
        else:
            fitness_func = opt_objective
        self.opt_obj = opt_objective
        self.fit = fitness_func
        self.loaded = True

    def _init_population(self):
        """Initialize the population"""
        # initialize container
        self.best_fit = None  # initialize the global best fitness
        self.best_idx = None  # initialize the global best solution's index
        self.best_ind = None  # initialize the global best solution
        self.pop_fit = np.full((self.N_pop), np.nan)  # initialize the container to store the fitness values of each individual
        self.pop_shel_idx = np.full((self.N_pop), -1, dtype=int)  # initialize the container to store the shelter indexes of each individual
        self.pop = np.full((self.N_pop, self.dim), 0.0)  # initialize the container to store each individual

        # initialize the population
        for i in range(self.N_pop):
            for j in range(self.dim):
                self.pop[i, j] = self.low_bound[j] + random.random() * self.diff_bound[j]
            # evaluate the fitness value
            fitness = self.fit(self.pop[i])
            self.pop_fit[i] = fitness
            # self.pop_shel_idx[i] = -1  # already full
            # calculate the best fitness so far
            if self.best_fit is None:
                self.best_fit = fitness
                self.best_idx = i
                self.best_ind = self.pop[i]
            else:
                if fitness < self.best_fit:
                    self.best_fit = fitness
                    self.best_idx = i
                    self.best_ind = self.pop[i]  

    def _init_shelters(self):
        """ Initialize the shelters """
        # initialize the container to store the sites of the shelters
        self.shel_sites = np.full((self.N_shels, self.dim), np.nan)
        self.shel_fit = np.full((self.N_shels), np.nan)

        # set the initial shelter leaders
        # sort the individuals by the ascendant order of their fitness values
        fit_ascendant_order_idx = np.argsort(self.pop_fit)
        self.shel_leaders_idx = fit_ascendant_order_idx[: self.N_shels]
        for shel_idx, leader_idx in enumerate(self.shel_leaders_idx):
            # `leader_idx` is the index of shelter leader in population
            self.pop_shel_idx[leader_idx] = shel_idx  # set the shelter index of the leaders
            self.shel_sites[shel_idx] = self.pop[leader_idx]  # set the shelter sites
            self.shel_fit[shel_idx] = self.pop_fit[leader_idx]  # set the fitness of shelter sites

        # set the base individual, C_base and fit_base, tobe the (N_shels + 1)th  individual of the sorted individuals
        self.base_idx = fit_ascendant_order_idx[self.N_shels + 1]
        self.base_individual = self.pop[self.base_idx]
        self.base_fitness = self.pop_fit[self.base_idx]

        # set the initial shelter followers
        pt = 0  # count for each shelters
        for i in range(self.N_shels):
            count = 1
            for k in range(pt, self.N_pop):
                if self.pop_shel_idx[k] < 0:  # TODO: check this later
                    self.pop_shel_idx[k] = i
                    count += 1
                    if count > self.cap:
                        pt = k
                        break

    def _quantified_shel(self, shel_idx):
        """Quantified by the normalized fitness value of the shelter site."""
        theta_s = self.base_fitness - self.shel_fit[shel_idx]
        theta_sum = float()
        for i in range(self.N_shels):
            theta_sum += self.base_fitness - self.shel_fit[i]
        return 1 - theta_s / theta_sum

    def _prob_leave(self, shel_idx):
        """For each exploit individual, it firstly evaluates its probability
        of leaving its current shelter based on probability model.
        This function calculate the probability for an individual to leave"""
        individuals_in_shel = 0
        for shel in self.pop_shel_idx:
            if shel == shel_idx:
                individuals_in_shel += 1
        return self._quantified_shel(shel_idx) / (1 + (individuals_in_shel / self.cap) ** 2)

    def _prob_enter(self, shel_idx):
        """For each explore individual, it randomly selects a shelter, 
        and then evaluates the probability of entering it.
        This function calculate the probability for an individual to enter."""
        individuals_in_shel = 0
        for shel in self.pop_shel_idx:
            if shel == shel_idx:
                individuals_in_shel += 1
        return (1 - self._quantified_shel(shel_idx)) * (1 - individuals_in_shel / self.cap)

    def _generalized_search(self, individual_idx):
        """If an individual is an explore individual, it will Perform the generalized search.
        Firstly, it randomly selects two different individuals to generate a mutant."""
        # two differ- ent individuals
        random_choose = random.sample(range(self.N_pop), 3)
        if individual_idx in random_choose:
            random_choose.remove(individual_idx)
            rand_idx_1, rand_idx_2 = random_choose
        else:
            rand_idx_1, rand_idx_2, _ = random_choose
        rand_indi_1 = self.pop[rand_idx_1]
        rand_indi_2 = self.pop[rand_idx_2]
        choosen_indi = self.pop[individual_idx]

        # generate mutation
        random_mut_1, random_mut_2 = np.random.rand(2, self.dim)
        mutation = choosen_indi + self.alpha * random_mut_1 * (rand_indi_1 - choosen_indi) + self.alpha * random_mut_2 * (rand_indi_2 - choosen_indi)

        # crossover
        r = random.randint(0, self.dim - 1)  # can ensure that at least one dimension can be altered
        new_indi = np.full((self.dim), 0.0)
        for j in range(self.dim):
            if random.random() < self.cr_lc or j == r:
                new_indi[j] = mutation[j]
            else:
                new_indi[j] = choosen_indi[j]

        # movement trials
        new_fitness = self.fit(new_indi)
        if new_fitness < self.pop_fit[individual_idx]:
            return new_indi, new_fitness
        else:
            return choosen_indi, self.pop_fit[individual_idx]

    def _local_search_follower(self, individual_idx):
        """If an individual is a follower in the sth shelter, it will
        move towards the shelter site"""
        shelter_site = self.shel_sites[self.pop_shel_idx[individual_idx]]
        choosen_indi = self.pop[individual_idx]

        # generate a mutation
        mutation = choosen_indi + 2 * random.random() * (shelter_site - choosen_indi)

        # crossover
        r = random.randint(0, self.dim - 1)  # can ensure that at least one dimension can be altered
        new_indi = np.full((self.dim), 0.0)
        for j in range(self.dim):
            if random.random() < self.cr_lc or j == r:
                new_indi[j] = mutation[j]
            else:
                new_indi[j] = choosen_indi[j]

        # movement trials
        new_fitness = self.fit(new_indi)
        if new_fitness < self.pop_fit[individual_idx]:
            return new_indi, new_fitness
        else:
            return choosen_indi, self.pop_fit[individual_idx]

    def _local_search_leader(self, individual_idx):
        """If an individual is a shelter leader, it will search its
        neighboring area"""
        # generate a mutation
        choosen_indi = self.pop[individual_idx]
        random_mut = (np.random.rand(self.dim) * 2 - np.full((self.dim), 1.0)) * self.delta
        mutation = choosen_indi * random_mut

        # crossover
        r = random.randint(0, self.dim - 1)  # can ensure that at least one dimension can be altered
        new_indi = np.full((self.dim), 0.0)
        for j in range(self.dim):
            if random.random() < self.cr_lc or j == r:
                new_indi[j] = mutation[j]
            else:
                new_indi[j] = choosen_indi[j]

        # movement trials
        new_fitness = self.fit(new_indi)
        if new_fitness < self.pop_fit[individual_idx]:
            return new_indi, new_fitness
        else:
            return choosen_indi, self.pop_fit[individual_idx]

    def _individual_migration(self):
        """Migration method implement the migration of individulas."""
        for i in range(self.N_pop):
            if self.pop_shel_idx[i] != -1:  # if it is an exploit individual
                Q_s = self._prob_leave(self.pop_shel_idx[i])  # probability of leave
                if random.random() < Q_s:  # If leave the shelter
                    new_ind, new_fit = self._generalized_search(i)
                    self.pop_shel_idx[i] = -1                        
                else:  # If do not leave
                    if i in self.shel_leaders_idx:  # If is a shelter leader
                        new_ind, new_fit = self._local_search_leader(i)
                    else:
                        new_ind, new_fit = self._local_search_follower(i)
            else:  # if it is an explore individual
                choosen_shel = random.randint(0, self.N_shels - 1)
                R_s = self._prob_enter(choosen_shel)
                if random.random() < R_s:  # if enter the shelter
                    new_ind, new_fit = self._local_search_follower(i)
                    self.pop_shel_idx[i] = choosen_shel
                else:  # if do not enter any shelter
                    new_ind, new_fit = self._generalized_search(i)
            self.pop[i] = new_ind
            self.pop_fit[i] = new_fit

    def _update_shelters(self):
        """Update shelters with the new fitness values"""
        # sort the individuals by the ascendant order of fitness values
        fit_ascendant_order_idx = np.argsort(self.pop_fit)
        self.shel_leaders_idx = fit_ascendant_order_idx[: self.N_shels]

        for shel_idx, leader_idx in enumerate(self.shel_leaders_idx):
            # `leader_idx` is the index of shelter leader in population
            self.pop_shel_idx[leader_idx] = shel_idx  # set the shelter index of the leaders
            self.shel_sites[shel_idx] = self.pop[leader_idx]  # set the shelter sites
            self.shel_fit[shel_idx] = self.pop_fit[leader_idx]  # set the fitness of shelter sites

        # set the base individual, C_base and fit_base, tobe the (N_shels + 1)th  individual of the sorted individuals
        self.base_idx = fit_ascendant_order_idx[self.N_shels + 1]
        self.base_individual = self.pop[self.base_idx]
        self.base_fitness = self.pop_fit[self.base_idx]

    def _is_termination(self):
        """NAA terminates when one of the following termination cri- teria is satisfied:
        
        a) The current best solution falls into the acceptance range.
        b) The pre-set maximum generation number `iteration` is reached."""
        if self.acpt_rnge is not None:
            if self.pop_fit[self.shel_leaders_idx[0]] < self.acpt_rnge:
                return True
            else:
                return False
        else:
            return False

    def _solution(self):
        """Give the solution in the current situation."""
        solution_idx = self.shel_leaders_idx[0]
        solution = self.pop[solution_idx]
        solution_fit = self.pop_fit[solution_idx]
        try:
            solution_value = self.opt_obj(solution)
        except:
            raise ValueError("Error when calculate opt problem.")
        return solution, solution_fit, solution_value

    def _overall_procudure(self):
        """OVERALL PROCEDURES OF NAA"""
        self._init_population()
        self._init_shelters()
        for j in range(self.iter_time):
            self._individual_migration()  # perform migration and search
            self._update_shelters()  # update the shelters
            if self._is_termination():  # if termination criterions are met
                return self._solution()
        return self._solution()

    def solve(self, prt=False, fit=False, value=True):
        """Solve the loaded problem. Problem loaded required.
        Args:
            prt   - Boolen, Print solution or not
            fit   - Boolen, Print fitness or not
            value - Boolen, Print solution value or not
        Return:
            array of solution."""
        assert self.loaded is True, "Optimization problem must be loaded into NAA before solved."
        solution, solution_fit, solution_value = self._overall_procudure()
        if prt:
            print("The solution:\n", end='')
            for number in solution:
                print("%.6f" % number, end=' ')
            if value:
                print(type(solution_value))
                print("\nThe solution value:\n\t%.8f" % solution_value)
            if fit:
                print("\nThe fitness value:\n\t%.8f" % solution_fit)
        return solution, solution_fit, solution_value


class NAA(NAA_base):

    def __init__(self, dimension, bound, iteration, parameters, verbose=None, acpt_rnge=None):
        super(NAA, self).__init__(dimension, bound, iteration, parameters, verbose, acpt_rnge)

