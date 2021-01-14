import numpy as np
import time
import sys
import random
import math
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

import pdb


class Hunting(tk.Tk, object):
    def __init__(self,
                 n_agents=5,
                 n_neighbors=2,
                 n_preys=20,
                 n_actions = 6,
                 unit = 15,
                 map_h = 20,
                 map_w = 20,
                 agent_size = 12,
                 prey_size = 12,
                 agent_pos = [],
                 prey_pos = [],
                 show = True
                 ):
        super(Hunting, self).__init__()
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.n_neighbors = n_neighbors
        self.n_preys = n_preys
        self.map_h = map_h
        self.map_w = map_w
        self.unit = unit
        self.agent_size = agent_size
        self.agent_pos_list = agent_pos
        self.agent_pos_list_his = agent_pos.copy()
        self.agent_v_list = np.zeros((self.n_agents,2))
        self.agent_list = []
        self.prey_size = prey_size
        self.prey_pos_list = prey_pos
        self.prey_pos_list_his = prey_pos.copy()
        self.prey_v_list = np.zeros((self.n_preys,2))
        self.prey_list = []
        self.n_preys_alive = self.n_preys
        self.done = False
        self.show = show
        if self.show:
            self.title('maze')
            self.geometry('{0}x{1}'.format(self.map_w * self.unit, self.map_h * self.unit))
        self._build_maze()


    def _build_maze(self,):
        for i in range(self.n_agents):
            if i*2 >= len(self.agent_pos_list):
                loc_h = random.randint(0, self.map_h-1)
                loc_w = random.randint(0, self.map_w-1)
                self.agent_pos_list.append(loc_w)
                self.agent_pos_list.append(loc_h)
            self.agent_v_list[i][0] = 0.
            self.agent_v_list[i][1] = 0.

        for i in range(self.n_preys):
            if i*2 >= len(self.prey_pos_list):
                loc_h = random.randint(0, self.map_h-1)
                loc_w = random.randint(0, self.map_w-1)
                self.prey_pos_list.append(loc_w)
                self.prey_pos_list.append(loc_h)
            self.prey_v_list[i][0] = 0.
            self.prey_v_list[i][1] = 0.

        if self.show:
            self.canvas = tk.Canvas(self, bg='white',
                            height=self.map_h * self.unit,
                            width=self.map_w * self.unit)

            # create origin
            origin = np.array([self.unit/2, self.unit/2])

            # create agents
            for i in range(self.n_agents):
                self.agent_list.append(self.canvas.create_oval(
                    origin[0] + self.unit*(self.agent_pos_list[i*2]) - self.agent_size/2, origin[1] + self.unit*(self.agent_pos_list[i*2+1]) - self.agent_size/2,
                    origin[0] + self.unit*(self.agent_pos_list[i*2]) + self.agent_size/2, origin[1] + self.unit*(self.agent_pos_list[i*2+1]) + self.agent_size/2,
                    fill='red'))

            # create preys
            for i in range(self.n_preys):
                self.prey_list.append(self.canvas.create_oval(
                    origin[0] + self.unit*(self.prey_pos_list[i*2]) - self.prey_size/2, origin[1] + self.unit*(self.prey_pos_list[i*2+1]) - self.prey_size/2,
                    origin[0] + self.unit*(self.prey_pos_list[i*2]) + self.prey_size/2, origin[1] + self.unit*(self.prey_pos_list[i*2+1]) + self.prey_size/2,
                    fill='blue'))

        # pack all
            self.canvas.pack()


    def reset(self):
        self.done = False
        self.agent_pos_list = self.agent_pos_list_his.copy()
        for i in range(self.n_agents):
            if i*2 >= len(self.agent_pos_list):
                loc_h = random.randint(0, self.map_h-1)
                loc_w = random.randint(0, self.map_w-1)
                self.agent_pos_list.append(loc_w)
                self.agent_pos_list.append(loc_h)
            self.agent_v_list[i][0] = 0.
            self.agent_v_list[i][1] = 0.

        self.prey_pos_list = self.prey_pos_list_his.copy()
        for i in range(self.n_preys):
            if i*2 >= len(self.prey_pos_list):
                loc_h = random.randint(0, self.map_h-1)
                loc_w = random.randint(0, self.map_w-1)
                self.prey_pos_list.append(loc_w)
                self.prey_pos_list.append(loc_h)
                self.prey_v_list[i][0] = 0.
                self.prey_v_list[i][1] = 0.

        if self.show:
            self.update()

            for i in range(self.n_agents):
                temp = self.agent_list.pop()
                self.canvas.delete(temp)
            for i in range(self.n_preys_alive):
                temp = self.prey_list.pop()
                self.canvas.delete(temp)

            origin = np.array([self.unit/2, self.unit/2])
            # create agents
            for i in range(self.n_agents):
                self.agent_list.append(self.canvas.create_oval(
                    origin[0] + self.unit*(self.agent_pos_list[i*2]) - self.agent_size/2, origin[1] + self.unit*(self.agent_pos_list[i*2+1]) - self.agent_size/2,
                    origin[0] + self.unit*(self.agent_pos_list[i*2]) + self.agent_size/2, origin[1] + self.unit*(self.agent_pos_list[i*2+1]) + self.agent_size/2,
                    fill='red'))

            # create preys
            for i in range(self.n_preys):
                self.prey_list.append(self.canvas.create_oval(
                    origin[0] + self.unit*(self.prey_pos_list[i*2]) - self.prey_size/2, origin[1] + self.unit*(self.prey_pos_list[i*2+1]) - self.prey_size/2,
                    origin[0] + self.unit*(self.prey_pos_list[i*2]) + self.prey_size/2, origin[1] + self.unit*(self.prey_pos_list[i*2+1]) + self.prey_size/2,
                    fill='blue'))

        self.n_preys_alive = self.n_preys
        # return observation
        ret_env = self.get_ret_env()
        return ret_env


    def get_ret_env(self):
        ret_env = []
        for i in range(self.n_agents):
            nearist_n_neighbors = np.array([],dtype=np.int32)
            min_dis = np.array([])
            for j in range(self.n_agents):
                if j != i:
                    dis_x = self.agent_pos_list[i*2]-self.agent_pos_list[j*2]
                    dis_y = self.agent_pos_list[i*2+1]-self.agent_pos_list[j*2+1]
                    dis = math.sqrt(dis_x*dis_x + dis_y*dis_y)
                    if len(min_dis) < self.n_neighbors:
                        min_dis = np.append(min_dis, dis)
                        nearist_n_neighbors = np.append(nearist_n_neighbors, j)
                    else:
                        max_dis = min_dis[0]
                        max_id = 0
                        for k in range(self.n_neighbors):
                            if max_dis < min_dis[k]:
                                max_dis = min_dis[k]
                                max_id = k
                        if dis < max_dis:
                            min_dis[max_id] = dis
                            nearist_n_neighbors[max_id] = j
            for j in range(self.n_preys_alive):
                dis_x = self.agent_pos_list[i*2]-self.prey_pos_list[j*2]
                dis_y = self.agent_pos_list[i*2+1]-self.prey_pos_list[j*2+1]
                dis = math.sqrt(dis_x*dis_x + dis_y*dis_y)
                if len(min_dis) < self.n_neighbors:
                    min_dis = np.append(min_dis, dis)
                    nearist_n_neighbors = np.append(nearist_n_neighbors, -j-1)
                else:
                    max_dis = min_dis[0]
                    max_id = 0
                    for k in range(self.n_neighbors):
                        if max_dis < min_dis[k]:
                            max_dis = min_dis[k]
                            max_id = k
                    if dis < max_dis:
                        min_dis[max_id] = dis
                        nearist_n_neighbors[max_id] = -j-1
            neighbor_env = np.array([])
            neighbor_env = np.append(neighbor_env,i)
            neighbor_env = np.append(neighbor_env,self.agent_pos_list[2*i])
            neighbor_env = np.append(neighbor_env,self.agent_pos_list[2*i+1])
            neighbor_env = np.append(neighbor_env,self.agent_v_list[i][0])
            neighbor_env = np.append(neighbor_env,self.agent_v_list[i][1])
            for k in range(self.n_neighbors):
                neighbor_env = np.append(neighbor_env,nearist_n_neighbors[k])
                if nearist_n_neighbors[k] > -1:
                    neighbor_env = np.append(neighbor_env,self.agent_pos_list[2*nearist_n_neighbors[k]])
                    neighbor_env = np.append(neighbor_env,self.agent_pos_list[2*nearist_n_neighbors[k]+1])
                    neighbor_env = np.append(neighbor_env,self.agent_v_list[nearist_n_neighbors[k]][0])
                    neighbor_env = np.append(neighbor_env,self.agent_v_list[nearist_n_neighbors[k]][1])
                else:
                    neighbor_env = np.append(neighbor_env,self.prey_pos_list[2*(-1-nearist_n_neighbors[k])])
                    neighbor_env = np.append(neighbor_env,self.prey_pos_list[2*(-1-nearist_n_neighbors[k])+1])
                    neighbor_env = np.append(neighbor_env,self.prey_v_list[(-1-nearist_n_neighbors[k])][0])
                    neighbor_env = np.append(neighbor_env,self.prey_v_list[(-1-nearist_n_neighbors[k])][1])
            ret_env.append(neighbor_env)
        return np.array(ret_env)


    def step(self, action):
        reward = -1
        for i in range(self.n_preys_alive):
            base_action = np.array([0., 0.])
            show_v = [0.,0.]
            base_action[0] = float(random.randint(-100, 100))/100.
            if random.randint(0, 1) == 1:
                base_action[1] = math.sqrt(1. - base_action[0]*base_action[0])
            else:
                base_action[1] = -math.sqrt(1. - base_action[0]*base_action[0])
            self.prey_v_list[i][0] = base_action[0]*0.3+self.prey_v_list[i][0]*0.7
            show_v[0] = self.prey_v_list[i][0]
            self.prey_pos_list[i*2] += self.prey_v_list[i][0]
            if self.prey_pos_list[i*2] > self.map_w-0.5:
                self.prey_pos_list[i*2] = self.prey_pos_list[i*2] - self.map_w
                show_v[0] = show_v[0] - self.map_w
            if self.prey_pos_list[i*2] < -0.5:
                self.prey_pos_list[i*2] = self.prey_pos_list[i*2] + self.map_w
                show_v[0] = show_v[0] + self.map_w
            self.prey_v_list[i][1] = base_action[1]*0.3+self.prey_v_list[i][1]*0.7
            show_v[1] = self.prey_v_list[i][1]
            self.prey_pos_list[i*2+1] += self.prey_v_list[i][1]
            if self.prey_pos_list[i*2+1] > self.map_h-0.5:
                self.prey_pos_list[i*2+1] = self.prey_pos_list[i*2+1] - self.map_h
                show_v[1] = show_v[1] - self.map_h
            if self.prey_pos_list[i*2+1] < -0.5:
                self.prey_pos_list[i*2+1] = self.prey_pos_list[i*2+1] + self.map_h
                show_v[1] = show_v[1] + self.map_h
            
            if self.show:
                self.canvas.move(self.prey_list[i], show_v[0]*self.unit, show_v[1]*self.unit)  # move prey

        for i in range(self.n_agents):
            base_action = np.array([0., 0.])
            show_v = [0.,0.]
            if action[i] == 0:   # up
                    base_action[1] -= 1
            elif action[i] == 1:   # down
                    base_action[1] += 1
            elif action[i] == 2:   # right
                    base_action[0] += 1
            elif action[i] == 3:   # left
                    base_action[0] -= 1
            elif action[i] == 4:
                j = 0
                while j < self.n_preys_alive:
                    x_dis = (self.prey_pos_list[2*j]-self.agent_pos_list[2*i])*(self.prey_pos_list[2*j]-self.agent_pos_list[2*i])
                    y_dis = (self.prey_pos_list[2*j+1]-self.agent_pos_list[2*i+1])*(self.prey_pos_list[2*j+1]-self.agent_pos_list[2*i+1])
                    d_dis = math.sqrt(x_dis+y_dis)
                    if d_dis < float(self.prey_size+self.agent_size)/(self.unit*2.):
                        self.delete_prey(j)
                        j -= 1
                        reward = reward + 10
                    j += 1
            self.agent_v_list[i][0] = base_action[0]*0.3+self.agent_v_list[i][0]*0.7
            show_v[0] = self.agent_v_list[i][0]
            self.agent_pos_list[i*2] += self.agent_v_list[i][0]
            if self.agent_pos_list[i*2] > self.map_w-0.5:
                self.agent_pos_list[i*2] = self.agent_pos_list[i*2] - self.map_w
                show_v[0] = show_v[0] - self.map_w
            if self.agent_pos_list[i*2] < -0.5:
                self.agent_pos_list[i*2] = self.agent_pos_list[i*2] + self.map_w
                show_v[0] = show_v[0] + self.map_w
            self.agent_v_list[i][1] = base_action[1]*0.3+self.agent_v_list[i][1]*0.7
            show_v[1] = self.agent_v_list[i][1]
            self.agent_pos_list[i*2+1] += self.agent_v_list[i][1]
            if self.agent_pos_list[i*2+1] > self.map_h-0.5:
                self.agent_pos_list[i*2+1] = self.agent_pos_list[i*2+1] - self.map_h
                show_v[1] = show_v[1] - self.map_h
            if self.agent_pos_list[i*2+1] < -0.5:
                self.agent_pos_list[i*2+1] = self.agent_pos_list[i*2+1] + self.map_h
                show_v[1] = show_v[1] + self.map_h
            
            if self.show:
                self.canvas.move(self.agent_list[i], show_v[0]*self.unit, show_v[1]*self.unit)  # move agent
            
        done = False
        if self.n_preys_alive == 0:
            reward += self.n_preys*10
            done = True
            
        if self.done:
            done_final = True
        else:
            done_final = False
        self.done = done

        ret_env = self.get_ret_env()
        return ret_env, reward, done_final


    def delete_prey(self, j):
        self.canvas.delete(self.prey_list[j])
        del self.prey_list[j]
        del self.prey_pos_list[j*2+1]
        del self.prey_pos_list[j*2]
        np.delete(self.prey_v_list,j)
        self.n_preys_alive -= 1


    def render(self):
        #time.sleep(0.01)
        if self.show:
            self.update()

    def get_n_features(self):
        return 5

    def get_act_space(self):
        return 6