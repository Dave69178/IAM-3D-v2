from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
from matplotlib.animation import FuncAnimation, FFMpegWriter
from Boid import Boid
from Environment import Environment

sns.set()


def initialise_boids(num_boids, random_velocity=False):
    boids = [Boid(i) for i in range(0, num_boids)]
    for boid in boids:
        # Choose random vector within sphere of radius 12
        distance_from_origin = 12 * np.random.uniform(0, 1)
        random_normal_3dvec = np.random.normal(0, 1, 3)
        boid.position = (12 * np.power(np.random.uniform(0, 1), 1 / 3) * random_normal_3dvec) / np.linalg.norm(
            random_normal_3dvec)

        # Get velocity and normalise
        if random_velocity:
            velocity = np.random.normal(0, 1, 3)
        else:
            velocity = np.array([1, 0, 0])
        boid.velocity = velocity / np.linalg.norm(velocity)
    return boids


def initialise_environment(boids):
    return Environment(boids)


def run_experiment():
    # Initialise list for storing position and velocity data
    positions = env.get_boid_positions()
    time = []
    x = []
    y = []
    z = []
    u = []
    v = []
    w = []
    for i in range(num_boids):
        time.append(0)
        x.append(positions[i, 0])
        y.append(positions[i, 1])
        z.append(positions[i, 2])
        u.append(positions[i, 0])
        v.append(positions[i, 1])
        w.append(positions[i, 2])

    # Run model and store data at each step
    for i in env.run(model, speed, time_step, max_angle_change_degrees, eta, fov, repulsion_range, orientation_metric_range, orientation_topological_range, attraction_topological_range, attraction_metric_range, max_time):
        positions = env.get_boid_positions()
        velocities = env.get_boid_velocities()
        for j in range(num_boids):
            time.append(env.time)
            x.append(positions[j, 0])
            y.append(positions[j, 1])
            z.append(positions[j, 2])
            u.append(velocities[j, 0])
            v.append(velocities[j, 1])
            w.append(velocities[j, 2])
    # Create and return dataframe with position and velocity data
    return pd.DataFrame({"time": time, "x": x, "y": y, "z": z, "u": u, "v": v, "w": w})


def save_df_as_csv(dataframe, name):
    dataframe.to_csv("./" + str(name) + ".csv")


def create_csv_for_cc_analysis(df, interval_num, name):
    # Creates csv containing boid data at specified intervals
    CC_dataset = df[(df["time"] % interval_num == 0)]
    CC_dataset.to_csv("./" + str(name) + ".csv")


def update_graph(num, axis, dataframe,):
    global Q
    Q.remove()
    data = dataframe[dataframe["time"] == num]
    #graph._offsets3d = (data.x, data.y, data.z)
    #min_val = min(min(data.x), min(data.y), min(data.z))
    #max_val = max(max(data.x), max(data.y), max(data.z))
    if num % 1 == 0:
        min_val_x = min(data.x) - 2
        max_val_x = max(data.x) + 2
        min_val_y = min(data.y) - 2
        max_val_y = max(data.y) + 2
        min_val_z = min(data.z) - 2
        max_val_z = max(data.z) + 2
        axis.set_xlim(min_val_x, max_val_x)
        axis.set_ylim(min_val_y, max_val_y)
        axis.set_zlim(min_val_z, max_val_z)
    axis.set_title('3D Test, time={}'.format(num))
    Q = axis.quiver(data.x, data.y, data.z, data.u, data.v, data.w)


def plot_boids(boid_df, save_animation=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    global Q
    Q = ax.quiver([], [], [], [], [], [])
    data = boid_df[boid_df["time"] == 0]
    anim = FuncAnimation(fig, partial(update_graph, axis=ax, dataframe=boid_df), max_time, interval=40, blit=False)
    if save_animation:
        f = r"./animation3.mp4"
        writervideo = FFMpegWriter(fps=60)
        anim.save(f, writer=writervideo)
    plt.show()


# Set parameters
num_boids = 100
random_starting_velocity = False
model = "topological"  # Enter string of name: ["metric", "topological", "hybrid"]
speed = 3
time_step = 0.1
max_angle_change_degrees = 4
eta = 0.1
fov = 300
repulsion_range = 1
orientation_metric_range = 5
orientation_topological_range = 5
attraction_topological_range = 15
attraction_metric_range = 10  # Upper limit, attraction range has lower bound orientation_metric_range
max_time = 500

# Initialise boids and environment
#boids = initialise_boids(num_boids, random_starting_velocity)
#env = initialise_environment(boids)
#output_df = run_experiment()
#plot_boids(output_df)

for i in range(0, 1):
    print(i)
    boids = initialise_boids(num_boids, random_starting_velocity)
    env = initialise_environment(boids)
    output_df = run_experiment()
    csv_name = f"df_test_{i}"
    #  create_csv_for_cc_analysis(output_df, 1, csv_name)
    #save_df_as_csv(output_df, csv_name)
    plot_boids(output_df)


