import sys
import numpy as np
import pandas as pd

np.set_printoptions(threshold=sys.maxsize)

print("Processing Data...")

np_train_data = np.empty([1, 2])

three_tracks = [4]

for i in three_tracks:

        pd_file = pd.read_csv("raw/data/" + str(i) + "_tracks.csv", skiprows=1, names=["frame", "id", "x", "y", "width", "height", "xVel", "yVel", "xAcc", "yAcc", "fSD", "bSD", "dhw", "thw", "ttc", "preXVel", "preId", "followId", "leftPreId", "leftAId", "leftFId", "rightPreId", "rightAId", "rightFId", "laneId"]);

        pd_file = pd_file[["x", "y"]]

        np_pd = np.array(pd_file, dtype="float32")

        np_train_data = np.vstack((np_train_data, np_pd))

#Calculate delta_x and delta_y
np_train_data = np.diff(np_train_data, axis=0)

#Remove elements > +/- 1 To ensure no big elements exist. 
#(during vehicle ID change, delta_x and delta_y will be huge) as trajectories will start elsewhere
np_train_data = np_train_data[np.logical_and(abs(np_train_data[:, 0]) < 1.0, abs(np_train_data[:, 1]) < 1.0), :]

print("Saving data...")

with open("raw/data/three_track_train_data.npy", "wb") as f:
    np.save(f, np_train_data)
