import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ResourceOptimizationEnv(gym.Env):
    """Synthetic optimization environment; never connects to a real host."""
    def __init__(self):
        super().__init__()
        self.observation_space=spaces.Box(0,1,shape=(5,),dtype=np.float32)
        self.action_space=spaces.Discrete(4)
        self.state=np.zeros(5,dtype=np.float32); self.steps=0
    def reset(self,seed=None,options=None):
        super().reset(seed=seed); self.steps=0
        self.state=np.array([self.np_random.uniform(.2,.85),self.np_random.uniform(.2,.85),
            self.np_random.uniform(.15,.8),self.np_random.uniform(.2,.9),self.np_random.uniform(.2,.8)],dtype=np.float32)
        return self.state.copy(),{}
    def step(self,action):
        if not self.action_space.contains(action): raise ValueError("Invalid action")
        cpu,mem,latency,workload,interval=self.state.tolist()
        cost=0.0
        if action==1:
            workload=max(.1,workload-.12); cpu=max(.05,cpu-.13); latency=max(.05,latency-.08); cost=.04
        elif action==2:
            cpu=max(.05,cpu-.12); mem=max(.05,mem-.08); latency=max(.05,latency-.14); cost=.12
        elif action==3:
            interval=max(.05,interval-.2); cost=.03
        noise=self.np_random.normal(0,.02,3)
        cpu=float(np.clip(cpu+noise[0],0,1)); mem=float(np.clip(mem+noise[1],0,1))
        latency=float(np.clip(latency+noise[2],0,1))
        self.state=np.array([cpu,mem,latency,float(workload),float(interval)],dtype=np.float32)
        reward=float(1-.30*cpu-.25*mem-.40*latency-cost)
        self.steps+=1; terminated=self.steps>=200
        return self.state.copy(),reward,terminated,False,{"intervention_cost":cost}

def action_name(action):
    return {0:"hold",1:"reduce_noncritical_workload",2:"increase_capacity_budget",3:"increase_observation_frequency"}[int(action)]
