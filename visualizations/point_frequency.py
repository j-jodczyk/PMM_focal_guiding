import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

# Path to the file
file_path = './samples_thread_7.txt'

# Read and parse 3D points
points = []
i = 0
with open(file_path, 'r') as f:
    for line in f:
        i += 1
        if (i == 1000000):
            break
        x, y, z = map(float, line.strip().split(' '))
        points.append((x, y, z))

# Count occurrences of each point
point_counts = Counter(points)

# Separate data for plotting
x = [p[0] for p in point_counts.keys()]
y = [p[1] for p in point_counts.keys()]
z = [p[2] for p in point_counts.keys()]
sizes = [count * 0.01 for count in point_counts.values()]  # Scale sizes for visibility

# Create 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x, y, z, s=sizes, alpha=0.6, edgecolors="k")

# Set labels and title
ax.set_title("3D Point Frequency Visualization")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
