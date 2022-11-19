import numpy as np 
from sklearn.decomposition import PCA



def get_target_sim_matrix(target_matrix):
    def Jaccard(matrix):
        matrix = np.mat(matrix)
        numerator = matrix * matrix.T
        denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
        return numerator / denominator
    sim_matrix = Jaccard(target_matrix)
    return sim_matrix

def get_pca_feature(feature,n_components):
    pca = PCA(n_components=n_components)  # PCA dimension
    pca.fit(feature)
    return pca.transform(feature)