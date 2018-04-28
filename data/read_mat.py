import scipy.io

def read_mat(path):
    mat = scipy.io.loadmat(path)
    nba_data = mat['nba_data']
    data_x = nba_data[:,:-1]
    data_y = nba_data[:,-1]
    return data_x, data_y