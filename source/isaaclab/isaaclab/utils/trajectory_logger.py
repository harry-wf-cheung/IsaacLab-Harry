import torch
## log the end effector coords
class TrajectoryLogger:
    def __init__(self, filename:str):
        self.filename = "docs/logs/traj" + filename + ".txt"
        self.x =[]
        self.y = []
        self.z = []

    def add_data(self, step: int,  ee_pos: torch):
        ee_pos = ee_pos.cpu().numpy()
       # print(ee_pos)
        with open(self.filename, "a") as f:
            f.write(str(step))
            f.write(' : ')
            for point in ee_pos:
                f.write(str(point))
                f.write(' : ')
            f.write(',\n')
