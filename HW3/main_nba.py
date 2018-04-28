import numpy as np 
from LogisticRegression import LogisticRegression
from data.read_mat import read_mat
from tools.smote import Smote
from imblearn.combine import SMOTEENN

def data_aug(data_x, data_y):
    minority_index = np.where(data_y==1)
    minority_x = data_x[minority_index]
    
    s = Smote(minority_x,N=1*100)
    new_data_x = s.over_sampling()

    new_data_y = np.ones((new_data_x.shape[0]))
    return new_data_x, new_data_y

def main():
    data_x, data_y = read_mat(path='./data/nbadata.mat')
    # import pdb; pdb.set_trace()
    # print('------- Newton Method------')
    sm = SMOTEENN()
    # syn_data_x, syn_data_y = sm.fit_sample(data_x, data_y)
    syn_data_x, syn_data_y = data_aug(data_x,data_y)
    mean = np.mean(data_x, axis=1)
    mean = np.expand_dims(mean, axis=1)

    std = np.std(data_x, axis=1)
    std = np.expand_dims(std, axis=1)
    data_x = (data_x - mean) / std
    # syn_data_x, syn_data_y = data_aug(data_x,data_y)
    # import pdb; pdb.set_trace()
    mean_syn = np.mean(syn_data_x, axis=1)
    mean_syn = np.expand_dims(mean_syn, axis=1)

    std_syn = np.std(syn_data_x, axis=1)
    std_syn = np.expand_dims(std_syn, axis=1)
    syn_data_x = (syn_data_x - mean_syn) / std_syn

    # import pdb; pdb.set_trace()
    model = LogisticRegression(x_data=np.vstack([data_x, syn_data_x]), y_data=np.hstack([data_y,syn_data_y]), original_x=data_x, original_y=data_y)

    # model = LogisticRegression(x_data=syn_data_x, y_data=syn_data_y, original_x=data_x, original_y=data_y)
    # model = LogisticRegression(x_data=data_x, y_data=data_y, original_x=data_x, original_y=data_y)
    # model.model_fit(method='NewtonMethod',lr=5e-2,error_bound=1e-1)
    # model.model_fit(method='GradientDescent',lr=1e-3, error_bound=1e-2)
    model.model_fit(method='BFGS',lr=1e-1, error_bound=1e-1)
    # model.inference(data_x, data_y)
    # model.model_fit(method='BFGS',lr=1e-1, error_bound=4e-1)
    # model.model_fit(method='BFGS',lr=1e-1, error_bound=3e-1)
   

if __name__ == '__main__':
    main()              