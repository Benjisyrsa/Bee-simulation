import math
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
import time

def MakePollinationMatrix(visitedFlowers, numberFlowers):
    rows, cols = numberFlowers,numberFlowers
    pollinationMatrix = [([0]*cols) for i in range(rows)]

    numberOfBees = len(visitedFlowers)
    for b in range(numberOfBees):
        beeVisits = visitedFlowers[b]
        for f in range(len(beeVisits)-1):
            flowerPollinated1 = beeVisits[f]
            flowerPollinated2 = beeVisits[f+1]
            pollinationMatrix[flowerPollinated1][flowerPollinated2] = 1
            pollinationMatrix[flowerPollinated2][flowerPollinated1] = 1

    return pollinationMatrix


def MakeDistanceMatrix(flowerLocations):
    flowerLocationsArray = np.array(flowerLocations)
    diffs = flowerLocationsArray[:, np.newaxis, :] - flowerLocationsArray[np.newaxis, :, :]
    distanceMatrix = np.sqrt(np.sum(diffs**2, axis=2))
    
    return distanceMatrix

def MakeBeeToFlowerDistanceVector(beeLocation, flowerLocations):
    beeLocationArray = np.array([beeLocation])
    flowerLocationsArray = np.array(flowerLocations)
    diffs = flowerLocationsArray[:, np.newaxis, :] - beeLocationArray[np.newaxis, :, :]
    distanceVector = np.sqrt(np.sum(diffs**2, axis=2))

    return distanceVector


def CalculatePollinationScore(distanceMatrix,pollinationMatrix,numberOfBees):
    numFlowers = len(distanceMatrix)
    pollinationScore = 0
    numPollinations = 0
    for f1 in range(numFlowers):
        for f2 in range(numFlowers):
            if f2 > f1:
                if pollinationMatrix[f1][f2] > 0:
                    numPollinations += 1
                    pollinationScore += distanceMatrix[f1][f2]

    # Normalize to the number of bees
    pollinationScore = pollinationScore / numberOfBees

    return numPollinations, pollinationScore


def Make_Flower_Patches(length,num_patches,avg_flowers_per_patch,patch_radius):
    random_seed = np.random.RandomState(random_seed_value)
    fewest_flowers_per_patch = np.ceil(0.7*avg_flowers_per_patch)
    most_flowers_per_patch = np.ceil(1.4*avg_flowers_per_patch)

    rand_x_patches = random_seed.uniform(low=0, high=length, size=(num_patches,))
    rand_y_patches = random_seed.uniform(low=0, high=length, size=(num_patches,))

    flowers_each_patch = np.random.randint(most_flowers_per_patch-fewest_flowers_per_patch, size=(num_patches)) + fewest_flowers_per_patch
    flowerCounter = 0

    for flower_index in range(num_patches):
        px = rand_x_patches[flower_index]
        py = rand_y_patches[flower_index]
        
        num_flowers_this_patch = flowers_each_patch[flower_index]
        patch_x_radius = patch_radius*random_seed.uniform(low=0.5, high=1.5)
        patch_y_radius = patch_radius*random_seed.uniform(low=0.5, high=1.5)
        flower_x_this_patch = random_seed.uniform(low=-1*patch_x_radius + px, high=patch_x_radius + px, size=(int(num_flowers_this_patch),))
        flower_y_this_patch = random_seed.uniform(low=-1*patch_y_radius + py, high=patch_y_radius + py, size=(int(num_flowers_this_patch),))

        if flower_index == 0:
            flowers_x = flower_x_this_patch
            flowers_y = flower_y_this_patch
        else:
            flowers_x = np.append(flowers_x,flower_x_this_patch)
            flowers_y = np.append(flowers_y,flower_y_this_patch)

    return flowers_x, flowers_y


class Flowers:
    def __init__(self, positions: np.ndarray):
        #indexes = same individual
        self.positions = positions  #nx2
        self.nectar_amounts = np.ones(len(self.positions))*flower_nectar_amount #nx1
        self.flowers = [] #nx1 

        for i, position in enumerate(positions):
            self.flowers.append(Flower(position, i))

    def get_position_vectors(self):
        return self.positions[:,0], self.positions[:,1]


class Flower:
    def __init__(self, position: np.ndarray, index):
        self._position = position
        self.index = index
        self.time_until_nectar_replenish = nectar_replenish_time


class Swarm:
    def __init__(self, positions: np.ndarray):
        # nx3
        self.positions = positions

        self.bees = []
        for i, position in enumerate(positions):
            self.bees.append(Bee(position))
        
    def update(self):
        self.positions[:,0] += np.cos(self.positions[:,2])*velocity*dt
        self.positions[:,1] += np.sin(self.positions[:,2])*velocity*dt


    def get_position_vectors(self):
        return self.positions[:,0], self.positions[:,1]


class Bee:
    def __init__(self, position: np.ndarray):
        self._position = position
        self.smell_list = []
        self.visited_flowers = []
        self.current_nectar = 0
        self.distance_until_patch = 0
        self.is_transporting = False

    def angle_towards(self, to_pos: list, use_std=False):
        dx = to_pos[0]-self._position[0]
        dy = to_pos[1]-self._position[1]
        angle = math.atan2(dy, dx)
        deviation = 0
        if use_std:
            deviation = np.random.normal(0, angle_deviation)

        self._position[2] = angle + deviation

    def collision_with_flower(self, j):
        if self.is_transporting:
            return
        
        if len(self.visited_flowers) > 0 and self.visited_flowers[-1] == j:
            return
        
        if len(self.visited_flowers) == 0:
            self.visited_flowers.append(j)

        if len(self.visited_flowers) > 0 and self.visited_flowers[-1] != j:
            self.visited_flowers.append(j)

        if flowers.nectar_amounts[j] > 0:
            self.current_nectar += flowers.nectar_amounts[j]
            flowers.nectar_amounts[j] = 0

        if self.current_nectar >= bee_nectar_storage:
            self.angle_towards(hive_position) 
            self.is_transporting = True
            return
        
        smell_index = random.randint(0, len(self.smell_list)-1)
        new_flower_index = self.smell_list[smell_index]
        self.angle_towards(flowers.positions[new_flower_index], True) 

    def update(self):
        #print(f"trans: {self.is_transporting}, nectar: {self.current_nectar}, smell: {len(self.smell_list)}")

        distance_from_hive = math.dist(self._position[:2], hive_position)

        if distance_from_hive <= collision_radius:
            self.returned_to_hive()
            return
        
        if self.is_transporting:
            if distance_from_hive >= self.distance_until_patch:
                if len(self.smell_list) == 0:
                    self.angle_towards(hive_position)
                    self.is_transporting = True
                else:
                    new_flower_index = self.smell_list[0]
                    self.angle_towards(flowers.positions[new_flower_index]) 
                    self.is_transporting = False

        if len(self.smell_list) == 0 and self.is_transporting == False:
            self.angle_towards(hive_position)
            self.is_transporting = True


    def returned_to_hive(self):
        self.current_nectar = 0
        flower_index = random.randint(0, len(flowers.positions)-1)
        self.angle_towards(flowers.positions[flower_index], True)
        self.distance_until_patch = math.dist(self._position[:2], flowers.positions[flower_index])
        self.is_transporting = True

def results_calculations():
    distanceMatrix = MakeDistanceMatrix(flowers.positions)
    visited_flowers_matrix = []
    for i, bee in enumerate(swarm.bees):
        bee: Bee
        visited_flowers_matrix.append(bee.visited_flowers)


    visited_flowers_matrix = np.array(list(itertools.zip_longest(*visited_flowers_matrix, fillvalue=0))).T
    pollination_matrix = MakePollinationMatrix(visited_flowers_matrix, len(flowers.positions))
    N_pollination_events, pollination_score = CalculatePollinationScore(distanceMatrix, pollination_matrix, N_bees)
    
    return pollination_score, N_pollination_events, pollination_matrix

#parameters
random_seed_value = 1234
plot_interval = 300
dt = 0.2 #s
velocity = 7.5 #m/s
N_bees = 20 #st
N_flowers = 100 #st
max_time = 1000 #s
field_size = 400 #m
half_field_size = int(field_size/2)
collision_radius = dt*velocity*1 #m smallest radius to check if collisions happen
flower_discoverability_radius_good = 20 #m, for good navigation
flower_discoverability_radius_bad = 5 #m, for good navigation
nectar_replenish_time = 120 #seconds until flowers get nectar again
flower_nectar_amount = 5 #nextar per flower mg?
bee_nectar_storage = 45 #max storage of nectar in ml: source: 30-60mg
angle_deviation_good = 0.1 # angle deviation for good navigation
angle_deviation_bad = 0.3 # angle deviation for poor navigation

#world generation
num_patches = 4
avg_flowers_per_patch = 25
patch_radius = 20
flower_x, flower_y = Make_Flower_Patches(field_size, num_patches, avg_flowers_per_patch, patch_radius)
hive_position = [half_field_size, half_field_size]

flowers_positions = np.column_stack((flower_x, flower_y))

#results lists
save_result_interval = 40
time_lists = []
total_pollinations_lists = []
pollination_scores_lists = []
last_pollination_matrix = []
#angle_deviations = [angle_deviation_good,angle_deviation_good,angle_deviation_bad,angle_deviation_bad]
#flower_discoverability_radii = [flower_discoverability_radius_good, flower_discoverability_radius_bad,flower_discoverability_radius_good, flower_discoverability_radius_bad]
#NavDescription = ["Good Direction, Good Sight/Smell","Good Direction, Poor Sight/Smell","Poor Direction, Good Sight/Smell","Poor Direction, Poor Sight/Smell" ]
angle_deviations = [angle_deviation_good]
flower_discoverability_radii = [flower_discoverability_radius_good]
NavDescription = ["Good Direction, Good Sight/Smell"]


for navSetting in range(len(angle_deviations)):
    
    #runtime variables
    simulation_time = 0
    simulation_samples = 0

    #plot initialization
    ax = plt.gca() #get current axis
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('xkcd:green')
    
    flowers = Flowers(flowers_positions)
    bees_positions = np.column_stack((np.ones(N_bees)*half_field_size, np.ones(N_bees)*half_field_size, np.random.random(N_bees)*np.pi*2))
    swarm = Swarm(bees_positions)
    
    time_list = []
    total_pollinations_list = []
    pollination_scores_list = []
    angle_deviation = angle_deviations[navSetting]
    flower_discoverability_radius = flower_discoverability_radii[navSetting]

    print("press crl+C to exit and plot")
    sleep_time = 0
    #run simulation
    try:
        while True:
            simulation_time += dt
            simulation_samples += 1

            if simulation_time > max_time: #exit loop
                time_lists.append(time_list)
                pollination_scores_lists.append(pollination_scores_list)
                total_pollinations_lists.append(pollination_scores_list)
                plt.clf()
                break

            if True: #collisions with for loop slow
                for i, bee in enumerate(swarm.bees):
                    bee: Bee
                    bee.update()
                    
                    bee.smell_list = []
                    distanceVector = MakeBeeToFlowerDistanceVector(bee._position[:2], flowers.positions)
                    bee.smell_list = np.where(distanceVector < flower_discoverability_radius)[0]

                    current_collisions = []
                    current_collisions = np.where(distanceVector < collision_radius)[0]
                    if len(current_collisions) == 0:
                        continue

                    collision_index = current_collisions[random.randint(0, len(current_collisions)-1)]
                    bee.collision_with_flower(collision_index)

            #replenish flower nectar linearly
            flowers.nectar_amounts[np.where(flowers.nectar_amounts < flower_nectar_amount)] += (1/nectar_replenish_time)*dt
            #update positions
            swarm.update()

            if simulation_samples%plot_interval==0:
                #plotting graphics
                plt.cla() #clear axis
                plt.title(f"Beeoids (time={simulation_time:.1f}s)")
                plt.xlabel("x [m]")
                plt.ylabel("y [m]")
                plt.xlim(0, field_size)
                plt.ylim(0, field_size)

                #nest
                plt.scatter([half_field_size], [half_field_size], c="yellow", s=100,marker = "H")

                #flowers
                x_vector, y_vector = flowers.get_position_vectors()
                plt.scatter(x_vector, y_vector, s=0.2+flowers.nectar_amounts/flower_nectar_amount*10, c="blue")
                
                #bees
                x_vector, y_vector = swarm.get_position_vectors()
                plt.scatter(x_vector, y_vector, s=10, c="yellow")

                plt.pause(0.001)


            if simulation_samples%save_result_interval == 0:
                current_score, current_num_pollinations, pollination_matrix = results_calculations()
                time_list.append(simulation_time)
                pollination_scores_list.append(current_score)
                total_pollinations_list.append(current_num_pollinations)
                last_pollination_matrix = pollination_matrix
                
                if False:
                    plt.cla()
                    plt.title(f"Pollination Matrix t={simulation_time:.1f}")
                    plt.xlabel("flower index")
                    plt.ylabel("flower index")
                    plt.imshow(pollination_matrix)
                    plt.pause(0.001)

            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        time_lists.append(time_list)
        pollination_scores_lists.append(pollination_scores_list)
        total_pollinations_lists.append(pollination_scores_list)

plt.close()
plt.title(f"Pollination Matrix, Score: {pollination_scores_lists[0][-1]:.0f}")
plt.xlabel("flower index")
plt.ylabel("flower index")
plt.imshow(pollination_matrix)
if False:
    plt.savefig(f"plots/patches{num_patches}flowersperpatch{avg_flowers_per_patch}")
plt.show()
        
plt.figure(1)
plt.close()
for navSetting in range(len(angle_deviations)):
    plt.plot(time_lists[navSetting], pollination_scores_lists[navSetting], label= NavDescription[navSetting])
plt.xlabel('Time (s)')
plt.ylabel('Pollination Success Score')
plt.title('Pollination Success Over Time for Varying Pollution Conditions')
plt.legend()
plt.show()





