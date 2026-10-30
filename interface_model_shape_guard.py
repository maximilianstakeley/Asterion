# In ReinforcementLearning/interface.py, immediately after:
#   self.model = PPO.load(str(path))
# add this guard:
shape = getattr(self.model.observation_space, "shape", None)
if shape != (10,):
    raise ValueError(
        f"Incompatible PPO observation shape {shape!r}; expected (10,). "
        "Retrain the model using the explicit-missingness state contract."
    )
# Keep the check after model load so incompatible older models fail at load time,
# before the first prediction or any policy decision.
