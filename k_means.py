import numpy as np

def init(X: np.array, K: int):
    n = X.shape[0]
    centroids = X[np.random.choice(n, K, replace=False)]
    return centroids

def assign_closest_centroid(X: np.array, centroids: np.array):
    # for each x[i], we compute ||x[i] - k[j]||^2 and assign label j that gives the shortest distance
    # ||x[i] - k[j]||^2 = ||x[i]||^2  +  ||k[j]||^2  -  2 x[i] dot k[j] 
    # We have X (nxd) and K (kxd)
    K = centroids

    X_squared = np.sum(X*X, axis = 1, keepdims = True)  #(n x 1) where nth row is ||x[i]||^2
    K_squared = np.sum(K*K, axis = 1, keepdims = True)  #(k x 1) where jth row is ||k[j]||^2
    X_dot_K = X @ K.T       #(n x k) where nth row X_dot_K[n,j] is x[i] dot k[j] 
    #                    (n x k) +  (n x 1)  +  (1 x k)    here nx1 and 1xk are broadcasted, becoming nx3
    distance_matrix = -2*X_dot_K + X_squared + K_squared.T 
    assigned_labels = np.argmin(distance_matrix, axis=1) # for each nth row, we take the smallest jth distance 
    return assigned_labels # (nx1)

def get_new_centroids(X: np.array, centroids: np.array, assigned_labels: np.array):
    d = X.shape[1]
    k_means = np.zeros((3, d))
    K = centroids.shape[0] 

    for i in range(K):
        index_list = np.where(assigned_labels == i)[0] 
        kth_cluster = X[index_list] #(mxd)
        n = kth_cluster.shape[0]
        k_means[i] = np.sum(kth_cluster, axis=0)/n

    return k_means #(kxd)

def K_means_algorithm(X: np.array, T: int, K: int):
    
    initialized_centroids = init(X, K)
    current_centroids = initialized_centroids
    for t in range(T): #T number of iterations

        assigned_labels = assign_closest_centroid(X, current_centroids)
        current_centroids = get_new_centroids(X, current_centroids, assigned_labels)
    
    return current_centroids

def centroid_probabilities(X: np.array, centroids: np.array):
    # for each x[i], we compute ||x[i] - k[j]||^2 and assign label j that gives the shortest distance
    # ||x[i] - k[j]||^2 = ||x[i]||^2  +  ||k[j]||^2  -  2 x[i] dot k[j] 
    # We have X (nxd) and K (kxd)
    K = centroids

    X_squared = np.sum(X*X, axis = 1, keepdims = True)  #(n x 1) where nth row is ||x[i]||^2
    K_squared = np.sum(K*K, axis = 1, keepdims = True)  #(k x 1) where jth row is ||k[j]||^2
    X_dot_K = X @ K.T       #(n x k) where nth row X_dot_K[n,j] is x[i] dot k[j] 
    #                    (n x k) +  (n x 1)  +  (1 x k)    here nx1 and 1xk are broadcasted, becoming nx3
    distance_matrix = -2*X_dot_K + X_squared + K_squared.T 
    # Convert distances to similarity scores
    scores = -distance_matrix

    # Numerical stability trick
    scores = scores - np.max(scores, axis=1, keepdims=True)

    # Softmax
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return probs





