import numpy as np 
from LogisticRegression import LogisticRegression
import matplotlib.pyplot as plt


def main():
    data_x = np.array([[0., 0.],
                       [2., 2.],
                       [2., 0.],
                       [3., 0.]])
    data_y = np.array([0,0,+1,+1,])
    # import pdb; pdb.set_trace()
    # print('------- Newton Method------')
    np.random.seed(0)
    model = LogisticRegression(x_data=data_x, y_data=data_y, original_x=data_x, original_y=data_y)
    model.model_fit(method='NewtonMethod',lr=4e-2, error_bound=5e-2)
    norm_list_1 = [np.linalg.norm(item) for item in model.gradient_history]
    model.model_fit(method='GradientDescent',lr=1e-1, error_bound=1e-2)
    norm_list_2 = [np.linalg.norm(item) for item in model.gradient_history]
    
    model.model_fit(method='BFGS',lr=1e-1, error_bound=1e-2)
    norm_list_3 = [np.linalg.norm(item) for item in model.gradient_history]

    # Plot Gradient Norm Curve
    plt.figure(figsize=(8,5))
    colors = ['r-', 'b-', 'g-']
    label_list = ['NetonMethod','NegativeGradient','BFGS']
    # import pdb ; pdb.set_trace()
    for i in range(3):
        lists =  eval('norm_list_{}'.format(i+1))  
        plt.plot(range(len(lists)), lists, colors[i], label=label_list[i])
    # i = 1
    # lists =  eval('norm_list_{}'.format(i+1))  
    # plt.plot(range(len(lists)), lists, colors[i], label=label_list[i])
    plt.title("Gradient Norm of Several Methods")
    plt.xlabel("Epochs")
    plt.ylabel("Norm")
    plt.legend(loc=0)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()              