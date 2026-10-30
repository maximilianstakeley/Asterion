import argparse
from stable_baselines3 import PPO
from .env import ResourceOptimizationEnv, action_name
from ..config import MODEL_DIR

def main():
    p=argparse.ArgumentParser(); p.add_argument("--episodes",type=int,default=10); a=p.parse_args()
    path=MODEL_DIR/"rl"/"resource_optimization_ppo.zip"
    if not path.exists(): p.error("Policy missing; run python -m hunterseeker.rl.train first.")
    if a.episodes<1: p.error("--episodes must be positive")
    env=ResourceOptimizationEnv(); model=PPO.load(str(path),env=env); returns=[]
    for ep in range(a.episodes):
        obs,_=env.reset(seed=100+ep); done=False; total=0.; counts={}
        while not done:
            act,_=model.predict(obs,deterministic=True)
            obs,reward,term,trunc,_=env.step(int(act)); done=term or trunc; total+=reward
            name=action_name(int(act)); counts[name]=counts.get(name,0)+1
        returns.append(total); print(f"episode={ep+1} return={total:.3f} actions={counts}")
    print(f"mean_return={sum(returns)/len(returns):.3f}")
if __name__=="__main__": main()
