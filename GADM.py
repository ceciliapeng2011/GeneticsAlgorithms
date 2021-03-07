'''
GADM, AGADM
An adaptive genetic algorithm with diversity-guided
mutation and its global convergence property
'''

from os import system
import sys
import random
import numpy
import math

#%% a=10

def handle_inputs(input_array):
    pop_size = int(input_array[0])
    tournament_size = int(input_array[1])
    mutation_rate = float(input_array[2])
    iter_number = int(input_array[3])
    bag_size = int(input_array[4])
    item_weights = list(map(int,input_array[5].split(',')))
    item_values = list(map(int,input_array[6].split(',')))
    return {'pop_size':pop_size,
            'tournament_size': tournament_size,
            'mutation_rate': mutation_rate,
            'iter_number': iter_number,
            'bag_size': bag_size,
            'item_values': item_values,
            'item_weights': item_weights,
        }

def get_inputs(inpts):
    file_name = ""
    for i in inpts:
        if '.txt' in i:
            file_name = i
            break
    if file_name:
        f = open(file_name)
    lines = []

    for line in f:
        lines.append(line.rstrip('\n'))

    return lines, file_name

def get_params():
    input_array, file_name = get_inputs(sys.argv)
    paramaters = handle_inputs(input_array)
    return paramaters, file_name


class IndividualGene:
    '''
    Individual Gene representation which includes chromosome and fitness value
    '''
    def __init__(self, chromosome, fitness_val):
        '''
        This is the basic initialization function.
        :param chromosome: chromosome of this individual gene.
        :param fitness_val: fitness value of this chromosome
        '''
        # initialize private members with input values
        self.chromosome = chromosome
        self.fitness_val = fitness_val

class StandardGA:
    '''
    Base class for genetic algorithm. this class will provide basic functions to generate populations, do selection,
    crossover, mutation and etc.
    '''
    def __init__(self, fitness_func=None, fitness_func_context=None, select_type="rank", mut_prob=0.05, mut_bits=1,
                 cross_prob=0.95, cross_points=1, elitism=True, tournament_size=None):
        '''
        This is the basic initialization function.
        :param fitness_func: function to compute fitness value for one chromosome
        :param fitness_func_context: context of fitness function
        :param select_type: parent selection type. could be one of three following values: "rank"(rank wheel selection);
        "roulette"(roulette wheel selection);"tournament". Default is "rank"
        :param mut_prob: mutation probability. default is 0.05
        :param mut_bits: bit number of mutation. default is 1.
        :param cross_prob: crossover probability. default is 0.95
        :param cross_points: cross over points. default is 1.
        :param elitism: enable elitism or not
        :param tournament_size: size of tournament in case the selection type is "tournament". default is none.
        '''
        # initialize private members with input values
        self.fitness_func = fitness_func
        self.fitness_func_context = fitness_func_context
        self.select_type = select_type
        self.mut_prob = mut_prob
        self.mut_bits = mut_bits
        self.cross_prob = cross_prob
        self.cross_points = cross_points
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.population = None
        self.best_chromosome = None
        self.best_fitness = -numpy.inf

        # Check the correctness of input parameters
        if self.fitness_func is None or \
            self.mut_prob < 0 or self.mut_prob > 1 or \
            self.mut_bits < 1 or \
            self.cross_prob < 0 or self.cross_prob > 1 or \
            self.cross_points < 1 or \
            self.select_type not in ["rank", "roulette", "tournament"] or \
            (self.select_type == 'tournament' and self.tournament_size is None) or \
            self.elitism not in [True, False]:
            raise ValueError('Invalid input parameters found')
        return

    def generate_binary_population(self, pop_size, chromosome_b_length):
        '''
        This function will generate a population in binary format according to the given population size and chromosome
        binary length.
        :param pop_size: size of population
        :param chromosome_b_length: binary length of chromosome
        '''
        self.population = []
        for i in range(pop_size):
            chromosome = []
            for j in range (chromosome_b_length):
                if random.uniform(0, 1) > 0.5:
                    chromosome.append(1)
                else:
                    chromosome.append(0)
            #chromesome = [random.randint(0, 1)for j in range(chromosome_b_length)]
            self.population.append(IndividualGene(chromosome, 0))
        self.population_size = pop_size
        return

    def evaluate_population(self):
        '''
        This function will evaluate the fitness of all the chromosomes within current population
        '''
        # Emumerate every individual gene within current populate and evaluate each's fitness
        fitness_sum = 0
        for gene in self.population:
            gene.fitness_val = self.fitness_func(gene.chromosome, self.fitness_func_context)
            fitness_sum += gene.fitness_val

        self.fitness_sum = fitness_sum
        return

    def sort_population(self):
        '''
        This function will do the sorting on all the chromosomes within current population
        '''
        self.population.sort(key=lambda x: x.fitness_val)

        # update best chromosome and fitness value
        if (self.population[-1].fitness_val > self.best_fitness):
            self.best_chromosome = self.population[-1].chromosome
            self.best_fitness = self.population[-1].fitness_val

        print("-----best fitness: %d" % self.best_fitness)
        return

    def individual_rank_select(self):
        '''
        This function will randomly pick up parent candidates from current population based on the rank possibility
        :return: return one selected parent
        '''
        rank_sum = numpy.cumsum(range(1, self.population_size + 1))[-1]
        rank_random = random.uniform(0, rank_sum)
        rank_list = list(range(self.population_size))

        # do sorting on the rank list
        rank_list.sort(key=lambda x: self.population[x].fitness_val)
        rank_sum = 0
        for i in range(self.population_size):
            rank_sum += i+1
            if rank_sum > rank_random:
                return self.population[rank_list[i]]

        return self.population[rank_list[-1]]


    def individual_roulette_select(self):
        '''
        This function will randomly pick up parent candidates from current population based on the fitness possibility
        :return: return one selected parent
        '''
        roulette_sum = self.fitness_sum
        roulette_random = random.uniform(0, roulette_sum)

        roulette_sum = 0
        for i in range(self.population_size):
            roulette_sum += self.population[i].fitness_val
            if roulette_sum > roulette_random:
                return self.population[i]

        return self.population[-1]

    def individual_tournament(self):
        '''
        This function will randomly pick up parent candidates from current population with the number of tournament size.
        Among them, only two of them will be returned
        :return: return one selected parent
        '''
        tournament_list = []
        if (self.tournament_size >= self.population_size):
            # if tournament size is larger or equal to population size, the tournament list is just the whole population list
            tournament_list = list(range(self.population_size))
        else:
            # pick up parent candidates from current population with the number of tournament size
            temp_random = random.randrange(0, self.population_size)
            tournament_list = [temp_random]

            for i in range(1, self.tournament_size):
                while temp_random in tournament_list:
                    temp_random = random.randrange(0, self.population_size)
                tournament_list.append(temp_random)

        # do sorting on the tournament list
        tournament_list.sort(key=lambda x: self.population[x].fitness_val)

        # return two candidates with highest fitness values
        return self.population[tournament_list[-1]]

    def select_parents(self):
        '''
        this function will select two parent lists based on the selection type configured
        :return: two parent lists selected
        '''
        parent_list=[]
        #
        if (self.select_type == 'rank'):
            for i in range(self.population_size):
                parent = self.individual_rank_select()
                parent_list.append(parent)
        elif (self.select_type == 'roulette'):
            for i in range(self.population_size):
                parent = self.individual_roulette_select()
                parent_list.append(parent)
        elif (self.select_type == 'tournament'):
            for i in range (self.population_size):
                parent = self.individual_tournament()
                parent_list.append(parent)

        return parent_list

    def crossover(self, parent_list):
        '''
        this function is to do cross-over with parent chromosome list and return new chromosome list

        :param parent_list: the list of parent chromosome
        :return: new chromosome list
        '''
        new_population = []

        for i in range(0, len(parent_list), 2):
            parent1 = parent_list[i]
            if (i + 1) >= len(parent_list):
                parent2 = parent_list[0]
            else:
                parent2 = parent_list[i + 1]

            chromosome_len = len(self.population[0].chromosome)

            if self.cross_points == 1:
                if random.uniform(0, 1) < self.cross_prob:
                    # generate one cross point
                    cross_point = random.randrange(1, chromosome_len-1)
                    # do cross over on two parent chromosome
                    new1 = parent1.chromosome[:cross_point]+parent2.chromosome[cross_point:]
                    new2 = parent2.chromosome[:cross_point]+parent1.chromosome[cross_point:]
                    # add new chromosome into new population
                    new_population.append(IndividualGene(new1, self.fitness_func(new1, self.fitness_func_context)))
                    new_population.append(IndividualGene(new2, self.fitness_func(new2, self.fitness_func_context)))
                else:
                    new_population.append(IndividualGene(parent1.chromosome, parent1.fitness_val))
                    new_population.append(IndividualGene(parent2.chromosome, parent2.fitness_val))
            elif(self.cross_points == 2):
                if random.uniform(0, 1) < self.cross_prob:
                    # generate two cross points
                    cross_point1 = random.randrange(1, chromosome_len-1)
                    cross_point2 = random.randrange(1, chromosome_len-1)

                    # compare two cross points
                    if (cross_point1 > cross_point2):
                        # do cross over on two parent chromosome
                        new1 = parent1.chromosome[:cross_point2]+parent2.chromosome[cross_point2:cross_point1]+parent1.chromosome[cross_point1:]
                        new2 = parent2.chromosome[:cross_point2]+parent1.chromosome[cross_point2:cross_point1]+parent2.chromosome[cross_point1:]
                    else:
                        # do cross over on two parent chromosome
                        new1 = parent1.chromosome[:cross_point1] + parent2.chromosome[cross_point1:cross_point2] + parent1.chromosome[cross_point2:]
                        new2 = parent2.chromosome[:cross_point1] + parent1.chromosome[cross_point1:cross_point2] + parent2.chromosome[cross_point2:]

                    # add new chromosome into new population
                    new_population.append(IndividualGene(new1, self.fitness_func(new1, self.fitness_func_context)))
                    new_population.append(IndividualGene(new2, self.fitness_func(new2, self.fitness_func_context)))
                else:
                    new_population.append(IndividualGene(parent1.chromosome, parent1.fitness_val))
                    new_population.append(IndividualGene(parent2.chromosome, parent2.fitness_val))
            else:
                # more than 2 cross points are not supported now
                raise ValueError('> 2 cross points are not supported')


        return new_population

    def mutation(self, new_population):
        '''
        This function is to do mutation on the input new chromosome list
        :param new_population: the list of new chromosome
        :return: new chromosome list after mutation
        '''
        print("mutation probability %f"%self.mut_prob)
        chromosome_len = len(self.population[0].chromosome)
        for i in range(self.population_size):
            for j in range(chromosome_len):
                if random.uniform(0, 1) < self.mut_prob:
                    if (new_population[i].chromosome[j] == 1):
                        new_population[i].chromosome[j] = 0
                    else:
                        new_population[i].chromosome[j] = 1

        return new_population

    def mating(self, new_population):
        '''
        This function is to combine new population with current population, sort and leave first ranked chromosomes
        :param new_population: the list of new chromosome
        '''
        pool_list = self.population+new_population
        pool_list.sort(key=lambda x: x.fitness_val, reverse=True)
        self.population = pool_list[:self.population_size]
        return
    
    def diversity_measure(self, values):
        '''
        This function is to measure diversity of a population.
        :param values: context for value calculation
        '''
        L = len(self.population[0].chromosome)
        
        def calc_pj(self, j):
            N = self.population_size            
            S_fabs = L #FIXME      
                        
            g_avg = 0.
            for i in range(N):                   
                if self.population[i].chromosome[j] == 1:
                    g_avg += values[j] 
            g_avg /= L
            
            pj = 0.
            for i in range(N):  
                if self.population[i].chromosome[j] == 1:
                    itm = values[j] - g_avg
                else:
                    itm = -g_avg                
                itm *= itm                     
                pj += itm
            
            pj = math.sqrt(pj)        
            pj *= 1/(S_fabs*N)

            print("pj - %d : %f : %f"%(j, pj, g_avg))
            #FIXME: how to avoid zero?
            if math.isclose(pj,0.,rel_tol=1e-5) :
                pj = sys.float_info.epsilon
            return pj
        
        d = 0.
        for j in range (L):
            pj = calc_pj(self, j)
            d += pj * math.log(pj)
        d = 1+ 1/d
        return d
    
    def mutation_diversity(self,  lambda1, lambda2, k5, k6, k7, values):
        '''
        This function is to do mutation on the input new chromosome list with diversity measurement.
        :param 0 < lambda1 < lambda2 < 1 , O<k6 <k5 < 1 , and k7 > 0 is almost equal to 0
        :param values: context for value calculation
        :return: update self.mut_prob        
        '''
        d = self.diversity_measure(values)
        print("diversity measurement %f"%d)
        if d < lambda1:
            self.mut_prob =  k5
        elif d < lambda2:
            self.mut_prob = k6
        else:
            self.mut_prob = k7

def knapsack_fitness_function(chromosome, context):
    '''
    This is the fitness function for 0-1 knapsack problem
    :param chromosome:current chromosome for fitness calculation
    :param context: context for fitness calculation
    :return: fitness value caculated
    '''
    params = context
    total_values = 0
    total_weights = 0

    for i, bit in enumerate(chromosome):
        if bit == 1:
            total_values += params.get('item_values')[i]
            total_weights += params.get('item_weights')[i]

    if total_weights > params.get('bag_size'):
        return 0
    else:
        return total_values

if __name__ == "__main__":
    """
    Getting parameters from the given text file. 
    """
    params, file_name = get_params()
    iter_number = params.get('iter_number')
    print(params)

    '''
    Get the item number
    '''
    item_num = len(params.get('item_values'))

    '''
    Initialize SGA algorithm context
    '''
    SGA = StandardGA(knapsack_fitness_function, params, "tournament", 0.05, 1, 0.95, 2, True, params.get('tournament_size'))

    '''
    Generate binary population with given binary chromosume length
    '''
    SGA.generate_binary_population(params.get('pop_size'), item_num)

    '''
    Do Evaluation and sorting on current population
    '''
    SGA.evaluate_population()
    SGA.sort_population()
     
    '''
    Selection
    '''
    parent_list = SGA.select_parents()
           
    for index in range(iter_number):
        print("generation: %d start"% index)

        '''
        Crossover
        '''
        next_population = SGA.crossover(parent_list)
        
        '''
        perform diversity-guided mutation
        A1 = 0 . 01, Az -----0. 25, ks = 0 . 60, k6 = 0.03, and kz --~ 0. 000 01
        '''
        SGA.mutation_diversity(0.01, 0.25, 0.60, 0.03, 0.00001, params.get('item_values'))
        next_population = SGA.mutation(next_population)

        '''
        Mating and update current population
        '''
        SGA.mating(next_population)
        
        '''
        Do Evaluation and sorting on current population
        '''
        SGA.evaluate_population()
        SGA.sort_population()

        '''
        Selection
        '''
        parent_list = SGA.select_parents()
        
        #print("generation: %d finish"%index)



# %%
