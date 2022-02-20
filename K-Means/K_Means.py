import matplotlib.pyplot as plt
import numpy as np
import sys

def cost(pixels, centroids, clusters, num_of_pixels):
    # summing the euclidean sum squared of all the pixels to their corresponding centroid
    dist_sum = 0
    for i, cluster in enumerate(clusters):
        for pixel_index in cluster:
            dist_sum += np.linalg.norm(pixels[pixel_index] - centroids[i]) ** 2
    # returning the average loss
    return dist_sum / num_of_pixels


# a class representing the K-Means algorithm
class K_Means:
    def __init__(self, k, max_iterations, pixels, centroids=None):
        self.k = k
        self.max_iterations = max_iterations
        self.pixels = pixels
        if centroids is None:
            # setting the algorithms centroids as k random centroids from the pixels:
            k_indices = np.random.choice(self.pixels.shape[0], self.k, replace=False)
            self.centroids = [self.pixels[i] for i in k_indices]
        else:
            self.centroids = centroids
        self.num_of_centroids = len(self.centroids)
        self.iterations = []
        self.cost = []
        # The clusters is a list that is composed out of K lists (each list is a cluster)
        # Each cluster is a list that holds all the indices of the pixels closest to it
        self.clusters = [[] for i in range(self.k)]

    def predict(self):
        for i in range(self.max_iterations):
            # update clusters
            self.update_clusters()
            # update centroids
            old_centroids = self.centroids
            self.centroids = self.update_centroids()
            # adding the iteration and cost to matching lists
            self.iterations.append(i)
            self.cost.append(cost(self.pixels, self.centroids, self.clusters, len(self.pixels)))
            # check if converged - if so, break.
            if self.is_converged(old_centroids):
                break
        # return the prediction
        return self.create_cluster_labels()

    def create_cluster_labels(self):
        # returning the labels of the pixels as an array
        labels = np.empty(len(self.pixels))
        for cluster_i, cluster in enumerate(self.clusters):
            for pixel_i in cluster:
                labels[pixel_i] = cluster_i
        return labels

    def update_clusters(self):
        # Initializing the clusters as a list of lists
        updated_clusters = [[] for i in range(self.k)]
        # For each index and pixel:
        for i, pixel in enumerate(self.pixels):
            # Find the closest centroid and get its index
            centroid_i = self.closest_centroid(pixel)
            # add the pixel's index to the correct cluster
            updated_clusters[centroid_i].append(i)
        # set self.clusters as the updated clusters
        self.clusters = updated_clusters

    def closest_centroid(self, pixel):
        # finding the distances from the pixel to all centroids
        dists = [np.linalg.norm(pixel - centroid) for centroid in self.centroids]
        # returning the centroid with the minimal distance from the pixel
        return np.argmin(dists)

    def update_centroids(self):
        new_centroids = np.zeros((self.k, len(self.pixels[0])))
        # For each cluster and its index:
        for cluster_i, cluster in enumerate(self.clusters):
            # find the new centroid (the cluster's mean)
            if len(self.pixels[cluster]) == 0:
                continue
            cluster_mean = np.mean(self.pixels[cluster], axis=0)
            # assign the new centroid as the clusters centroid
            new_centroids[cluster_i] = cluster_mean

        # round all the centroids
        new_centroids = np.round(new_centroids, 4)
        return new_centroids

    def is_converged(self, old_centroids):
        # finding the distances from the old centroids to the centroids
        dists = [np.linalg.norm(old_centroids[i] - self.centroids[i]) for i in range(self.k)]
        # if the distances sum is 0, then it converged otherwise return false
        return sum(dists) == 0


if __name__ == '__main__':

    image_fname = sys.argv[1]

    # Setting "pixels" as a list of the pixels in the image
    orig_pixels = plt.imread(image_fname)
    pixels = orig_pixels.astype(float) / 255.
    # Reshape the image into an Nx3 matrix where N = number of pixels.
    pixels = pixels.reshape(-1, 3)

    centroids_fname = sys.argv[2]
    if centroids_fname.isdigit():
        centroids = None
        number_of_centroids = int(centroids_fname)
    else:
        # loading centroids
        centroids = np.loadtxt(centroids_fname)
        number_of_centroids = len(centroids)

    # Creating the prediction of the algorithm
    k_means = K_Means(number_of_centroids, 20, pixels, centroids)
    prediction = k_means.predict()

    # Creating the new image:
    new_pixels = [0 for i in range(len(pixels))]
    for i in range(len(prediction)):
        new_pixels[i] = k_means.centroids[prediction[i].astype(int)]
    new_pixels = np.array(new_pixels).reshape(orig_pixels.shape)
    new_file_name = f'compressed_with_{number_of_centroids}_colors.jpeg'
    plt.imsave(new_file_name, new_pixels)

    # Creating the graph - iterations and average loss function:
    # plt.xlabel('Iteration')
    # plt.ylabel('Average Loss')
    # plt.title('K = ')
    # plt.plot(k_means.iterations, k_means.cost)
    # plt.show()
