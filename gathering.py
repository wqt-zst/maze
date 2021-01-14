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


class Gathering(tk.Tk, object):
    def __init__(self,
                 n_agents=4,
                 n_neighbors=3,
                 n_count = 2,
                 map_h = 20,
                 map_w = 20,
                 page_h = 800,
                 page_w = 1000,
                 agent_pos = [],
                 show = True
                 ):
        super(Gathering, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_agents = n_agents
        self.n_neighbors = n_neighbors
        self.n_count = n_count
        self.map_h = map_h
        self.map_w = map_w
        unit_h = page_h/self.map_h
        unit_w = page_w/self.map_w
        if unit_h >unit_w:
            self.unit = int(unit_w)
        else:
            self.unit = int(unit_h)
        self.agent_size = int(0.8*self.unit)
        self.agent_pos_list = agent_pos
        self.agent_pos_list_his = agent_pos.copy()
        self.agent_v_list = np.zeros((self.n_agents,2))
        self.agent_list = []
        self.last_dis = 0
        self.done = False
        self.show = show
        if self.show:
            self.title('Gathering')
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

        # pack all
            self.canvas.pack()


    def reset(self):
        self.last_dis = 0
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

        if self.show:
            self.update()
            for i in range(self.n_agents):
                temp = self.agent_list.pop()
                self.canvas.delete(temp)

            origin = np.array([self.unit/2, self.unit/2])

            # create agents
            for i in range(self.n_agents):
                self.agent_list.append(self.canvas.create_oval(
                    origin[0] + self.unit*(self.agent_pos_list[i*2]) - self.agent_size/2, origin[1] + self.unit*(self.agent_pos_list[i*2+1]) - self.agent_size/2,
                    origin[0] + self.unit*(self.agent_pos_list[i*2]) + self.agent_size/2, origin[1] + self.unit*(self.agent_pos_list[i*2+1]) + self.agent_size/2,
                    fill='red'))

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
            neighbor_env = np.array([])
            neighbor_env = np.append(neighbor_env,i)
            neighbor_env = np.append(neighbor_env,self.agent_pos_list[2*i])
            neighbor_env = np.append(neighbor_env,self.agent_pos_list[2*i+1])
            neighbor_env = np.append(neighbor_env,self.agent_v_list[i][0])
            neighbor_env = np.append(neighbor_env,self.agent_v_list[i][1])
            for k in range(self.n_neighbors):
                neighbor_env = np.append(neighbor_env,nearist_n_neighbors[k])
                neighbor_env = np.append(neighbor_env,self.agent_pos_list[2*nearist_n_neighbors[k]])
                neighbor_env = np.append(neighbor_env,self.agent_pos_list[2*nearist_n_neighbors[k]+1])
                neighbor_env = np.append(neighbor_env,self.agent_v_list[nearist_n_neighbors[k]][0])
                neighbor_env = np.append(neighbor_env,self.agent_v_list[nearist_n_neighbors[k]][1])
            ret_env.append(neighbor_env)
        return np.array(ret_env)


    def step(self, action):
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
            
        # reward function

        dis_ = 0
        done = True
        for i in range(self.n_agents):
            max_dis_list = []
            for j in range(self.n_agents):
                if i != j:
                    dis_x = (self.agent_pos_list[i*2]-self.agent_pos_list[j*2])*(self.agent_pos_list[i*2]-self.agent_pos_list[j*2])
                    dis_y = (self.agent_pos_list[i*2+1]-self.agent_pos_list[j*2+1])*(self.agent_pos_list[i*2+1]-self.agent_pos_list[j*2+1])
                    dis = math.sqrt(dis_x + dis_y)
                    if len(max_dis_list) < self.n_count:
                        max_dis_list.append(dis)
                    else:
                        min_dis = max_dis_list[0]
                        min_dis_id = 0
                        for m in range(len(max_dis_list)):
                            if min_dis > max_dis_list[m]:
                                min_dis = max_dis_list[m]
                                min_dis_id = m
                        if dis > min_dis:
                            max_dis_list[min_dis_id] = dis
            if sum(max_dis_list) > 0.1:
                done = False
            dis_ = dis_ + sum(max_dis_list)
        reward = - dis_
        self.last_dis = 0

        if done:
            reward = self.n_agents*3*(self.map_h+self.map_w)/4

        if self.done:
            done_final = True
        else:
            done_final = False
        self.done = done

        ret_env = self.get_ret_env()

        return ret_env, reward, done_final

    def render(self):
        if self.show:
            self.update()

    def get_n_features(self):
        return 5

    def get_act_space(self):
        return 5