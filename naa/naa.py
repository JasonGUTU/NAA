# -*- coding: utf-8 -*-
"""
Author : JasonGUTU
Email  : hellojasongt@gmail.com
Python : anaconda3
"""
import random

import numpy as np
import scipy as sp


class NAA_base(object):

    def __init__(self, dimension, bound, iteration, parameters, verbose):
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
                             (4) Cr_local         - Local crossover factor,                      TODO: type
                             (5) Cr_global        - Global crossover factor,
                             (6) alpha            - Movement factor]
             verbose    - iteration information display flag.                                    TODO: verbose
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
        self.cr_lc = parameters[3]  # Local crossover factor
        self.cr_gb = parameters[4]  # Global crossover factor
        self.alpha = parameters[5]  # Movement factor

    def load_problem(opt_objective, fitness_func):
        """Load optimization problem and fitness function.

        Args:
            opt_objective - type: function.
            fitness_func  - type: function.
        """
        assert hasattr(opt_objective, '__call__'), "`opt_objective` must be callable."
        assert hasattr(fitness_func, '__call__'), "`fitness_func` must be callable."
        self.opt_obj = opt_objective
        self.fit = fitness_func

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
        shel_leader_site_idx = fit_ascendant_order_idx[: self.N_shels]
        for shel_idx, leader_idx in enumerate(shel_leader_site_idx):
            # `leader_idx` is the index of shelter leader in population
            self.pop_shel_idx[leader_idx] = shel_idx  # set the shelter index of the leaders
            self.shel_sites[shel_idx] = self.pop[leader_idx]  # set the shelter sites
            self.shel_fit[shel_idx] = self.pop_fit[leader_idx]  # set the fitness of shelter sites

        # set the base individual, C_base and fit_base, tobe the (N_shels + 1)th  individual of the sorted individuals
        self.base_idx = fit_ascendant_order_idx[self.N_shels + 1]
        self.base_individual = self.pop[base_idx]
        self.base_fitness = self.pop_fit[base_idx]

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
        

    def _individual_migration(self):
        """Migration method implement the migration of individulas."""
        for i in range(self.N_pop):
            if self.pop_shel_idx[i] != -1:  # if it is an exploit individual
                Q_s = self._prob_leave(self.pop_shel_idx[i])  # probability of leave
                if random.random() < Q_s:
                    
                    


            