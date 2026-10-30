import argparse
from stable_baselines3 import PPO
from .env import ResourceOptimizationEnv
from ..config import MODEL_DIR

def main():
    p=argparse.ArgumentParser(); p.add_argument("--timesteps",type=int,default=20000); a=p.parse_args()
    if a.timesteps<1000: p.error("--timesteps should be at least 1000")
    out=MODEL_DIR/"rl"; out.mkdir(parents=True,exist_ok=True)
    model=PPO("MlpPolicy",ResourceOptimizationEnv(),verbose=1,seed=42,n_steps=512,batch_size=64)
    model.learn(total_timesteps=a.timesteps); model.save(str(out/"resource_optimization_ppo"))
    print(f"Saved simulated policy to {out/'resource_optimization_ppo.zip'}; not deployed to live systems.")
if __name__=="__main__": main()
