import csv
import os
import matplotlib.pyplot as plt
import numpy as np

def read_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append([float(row[0]), float(row[1])])
    return np.array(data)

def EuclideanDistance(a, b):
    d = np.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)
    return d

def plot_points_with_lines(data1, data2):
    plt.scatter(data1[:, 0], data1[:, 1], label='Dataset 1', marker='.', color='blue')
    plt.scatter(data2[:, 0], data2[:, 1], label='Dataset 2', marker='x', alpha=0.8, color='green')
    x=[]
    num_points = min(len(data1), len(data2))
    for i in range(num_points):
        distance = EuclideanDistance(data1[i], data2[i])
        x.append(distance)
        if distance < 0.1:    
            plt.plot([data1[i, 0], data2[i, 0]], [data1[i, 1], data2[i, 1]], color='dimgrey', linewidth=0.4, alpha=0.7)
    #plt.hist(x, bins=20)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Points with Lines (Distance < 0.1)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Replace these file paths with your actual CSV file paths
    f = os.getcwd()
    csv_file_path2 = f + '/predictions/OnsDen_fW.csv'
    csv_file_path1 = f + '/embeddings/MDS.csv'

    data1 = read_csv(csv_file_path1)
    data2 = read_csv(csv_file_path2)
    data1 = np.array(data1, dtype=float)
    data2 = np.array(data2, dtype=float)
    # Combine the two datasets
    #combined_data = np.concatenate((data1, data2), axis=0)

    # Plot points with lines
    plot_points_with_lines(data1, data2)