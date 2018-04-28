import numpy as np 
# from optimizer import NewtonMethod, GradientDescent, BFGS
from tqdm import tqdm
class LogisticRegression(object):

    def __init__(self,x_data, y_data, original_x, original_y):
        self.optimizer = ''
        
        self.num_samples = x_data.shape[0]
        assert self.num_samples > 0
        # self.dim_feature = x_data.shape[1]
        # Add bias term
        self.x_data = np.hstack((x_data, np.ones((self.num_samples, 1))))
        self.y_data = y_data
        self.original_x = original_x
        self.original_y = original_y
        self.loss_mask = np.ones_like(self.y_data)#*1.5 + 0.5
        # dim_feature = x_feature_dim + 1
        self.dim_feature = self.x_data.shape[1]
        self.best_acc = 0
       
        # self.weights = np.ones((1,self.dim_feature))
        self.lambda_value = 200 # L2 regularization 
    def reset_weights(self,seed=0):
        np.random.seed(seed)
        self.weights = np.random.randn(1,self.dim_feature)

    def bfgs_init(self, rho=0.66, c1=0.1, c2=0.9):
        self.rho = 0.66
        self.c1 = 1e-4 # c1 in (0, 1)
        self.c2 = 0.9 # c2 in (c1,1)
        # Initial Eye Matrix
        self.Bk = np.eye(self.dim_feature)

    def model(self):
        pass
    def sigmoid(self, inputs):
        """
        \[\sigma(inputs) = \frac{\exp(inputs)}{1+\exp(inputs)}\]
        """
        # import pdb ; pdb.set_trace()
        sigmoid = np.exp(inputs) / (1.0 + np.exp(inputs))
        return sigmoid

    def sigmoid_derive(self,sigmoid):
        """
        Compute Sigmoid Derive
        """
        sigmoid_derive = sigmoid * (1 - sigmoid + 1e-10) 
        return sigmoid_derive

    def calculate_hessian(self, x_data, weights):

        A = np.copy(x_data)
        B = np.identity(self.num_samples) * self.sigmoid_derive(weights.dot(x_data.T))

        # hessian = self.sigmoid_derive(weights.dot(x_data.T)
        hessian = np.dot(np.dot(A.T, B), A)
        # hessian_inv = np.linalg.inv(hessian + 
        #                 self.lambda_value * np.eye(self.weights.shape[1]))  # 3x3
        hessian_inv = np.linalg.inv(hessian)  # 3x3
        # import pdb; pdb.set_trace()
        return hessian, hessian_inv

    def log_loss(self, weights):
        sigmoid_out = self.sigmoid(weights.dot(self.x_data.T))

        log_loss = np.mean(self.loss_mask*(-np.log(sigmoid_out)*self.y_data.T  - np.log(1-sigmoid_out)*(1-self.y_data.T)))
        # import pdb; pdb.set_trace()
        return log_loss

    def set_optimizer(self, mode='NewtonMethod'):


        optimizer = eval(mode)

    def loss_gradient(self, weights):
        score_diff = -self.sigmoid(weights.dot(self.x_data.T)) + self.y_data # (1, num_samples)
        gradient = (self.loss_mask*score_diff).dot(self.x_data) #(1, feature_dim)
        # import pdb; pdb.set_trace()
        return gradient

    def model_fit(self, method='gradient_descent', lr=1e-2, error_bound=1e-2,max_iters=1000000):
        """
        Args:
            method:
                `NewtonMethod`: newton method
                `GradientDescent`: Gradient Descent
                `BFGS`: quai-newton mthod: BFGS 
        """
        self.reset_weights()

        self.gradient_history = []
        self.loss_history = []

        iter_idx = 0

        # Choose one method
        if method == 'GradientDescent':
            model = self.gradient_descent
        elif method == 'NewtonMethod':
            model = self.newton_method
        elif method == 'BFGS':
            # Init some parameters
            self.bfgs_init()
            model = self.bfgs_method
        else:
            raise NotImplementedError
        print('-------------Use {} optmization method-----------'.format(method))

        # Start to optimize
        for iter_idx in range(max_iters):
            loss, loss_gradient=model(lr)
            # if self.log_loss(self.weights) < error_bound:
            #     break
            gradient_norm = np.linalg.norm(self.loss_gradient(self.weights))
            if gradient_norm < error_bound:
                break
            # iter_idx += 1
            if iter_idx % 50 == 0:
                print('Iterations:',iter_idx,'Loss:', loss, 'Weight Norm:', gradient_norm,)
                self.inference(self.original_x, self.original_y)
        print('Total Iterations:',iter_idx, 'Current Loss:', self.log_loss(self.weights))
       
        predicts = self.sigmoid(self.weights.dot(self.x_data.T))
        # print('Predict Probability:', predicts)
        # print('Predict Results:', np.round(predicts))

        acc = self.metric(np.round(predicts), self.y_data)
        print('Training Acc:',acc)

    def gradient_descent(self, lr):           
        loss = self.log_loss(self.weights) 
        loss_gradient = self.loss_gradient(self.weights)
        self.weights = self.weights + lr*loss_gradient

        self.gradient_history.append(loss_gradient)
        self.loss_history.append(loss)
        return loss, loss_gradient

    def newton_method(self, lr):
        # Compute Hessian Matrix
        hessian, hessian_inv = self.calculate_hessian(self.x_data, self.weights)

        # Loss
        loss = self.log_loss(self.weights)
        # Get gradient
        loss_gradient = self.loss_gradient(self.weights) #(1, feature_dim)

        step = np.dot(hessian_inv, loss_gradient.T).T #(1,feature_dim)
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        self.weights = self.weights - lr * step
        
        self.gradient_history.append(loss_gradient)
        self.loss_history.append(loss)
        return loss, loss_gradient

    def bfgs_method(self, lr=0):

        # Get current loss
        loss = self.log_loss(self.weights)
        # Get gradient, g_k
        loss_gradient = self.loss_gradient(self.weights)
        
        pk = 1. * np.linalg.solve(self.Bk, loss_gradient.T) 
        mk = self.wolfe_search(pk, loss, loss_gradient )
        
        # Get step length
        alpha = self.rho**mk
        
        # BFGS Correct
        # sigma_k =  x_{k+1} - x_k
        sigma_k = alpha * pk
        self.weights = self.weights + sigma_k.T

        # y_k = g_k+1 - g_k 
        yk = self.loss_gradient(self.weights) - loss_gradient
        
        # Update Bk
        if np.dot(yk,sigma_k) > 0:
            # Pk = y_k  y_K^T / y_k^T sigma_k
            Pk = 1.* np.dot(yk, yk.T) / (np.dot(yk, sigma_k) + 1e-16)
            Qk = -1.0 * np.dot(self.Bk,sigma_k).dot(sigma_k.T).dot(self.Bk) / (np.dot(sigma_k.T, self.Bk).dot(sigma_k) + 1e-16)
            self.Bk = self.Bk + Pk +Qk
        self.gradient_history.append(loss_gradient)
        self.loss_history.append(loss)
        return loss, loss_gradient

    def wolfe_search(self, pk,loss,loss_gradient):
        m = 1; mk = 0
        while m < 200:
            # use Wolfe condition to search the step
            weights_temp = np.copy(self.weights) + (self.rho**m)*pk.T 
            loss_gradient_temp = self.loss_gradient(weights_temp)
            loss_threshold = loss+self.c1 * (self.rho**m) * np.dot(loss_gradient, pk) 
                
                # print(self.log_loss(weights_temp))
            if self.log_loss(weights_temp) < loss_threshold \
                and np.dot(loss_gradient_temp, pk) >= self.c2*np.dot(loss_gradient, pk):
                mk = m
                break
            m += 1
        return mk 

    def metric(self, predicts, data_y):
        results = np.sum(np.abs(data_y - predicts))
        acc = 1- results /data_y.shape[0]
        return acc
        
    
    def inference(self, data_x, data_y):
        data_x = np.hstack((data_x, np.ones((data_y.shape[0], 1))))
        predicts = self.sigmoid(self.weights.dot(data_x.T))
        acc = self.metric(np.round(predicts), data_y)
        if acc > self.best_acc:
            self.best_acc = acc
        print('Accuracy on Original Dataset:', acc, 'Best Acc:', self.best_acc)
        