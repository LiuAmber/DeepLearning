import numpy as np
import matplotlib.pyplot as plt

def data(file_path):
    data_train = []
    with open(file_path, "r") as f:
        data = f.readlines()
        for line in data:
            numbers = line.split(",")
            numbers_float = list(map(float, numbers))
            data_train.append(numbers_float)

    data_train = np.array(data_train)
    x = []
    y = []
    for line in data_train:
        data_x = line[:1000]
        data_y = line[-1]
        x.append(data_x)
        y.append(data_y)
    x = np.array(x)
    y = np.array(y)
    y = np.reshape(y,(y.shape[0],1))
    x = x.T
    y = y.T
    return x,y

def batch_data(x,y,batchsize):
    train_data = list(zip(x,y))
    np.random.shuffle(train_data)
    x[:], y[:] = zip(*train_data)
    x = x.T
    y = y.T
    x_batch = x[:batchsize].T
    y_batch = y[:batchsize].T
    return x_batch,y_batch

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A,cache

def relu(Z):
    A = np.maximum(0,Z)
    cache = Z
    return A,cache



def relu_backward(dA, cache):
    Z = cache
    dz = np.array(dA,copy = True)
    dz[Z<=0] = 0
    return dz


def sigmoid_backward(dA, cache):
    z = cache
    s = 1/(1+np.exp(-z))
    dz = dA*s*(1-s)
    return dz

def init_parameter(layers_dim):
    np.random.seed(3)
    parameters = {}

    for i in range(1,len(layers_dim)):
        #parameters['w'+str(i)] = np.random.randn(layers_dim[i],layers_dim[i-1])*0.01
        parameters['w' + str(i)] = np.random.randn(layers_dim[i], layers_dim[i - 1]) *np.sqrt(2/layers_dim[i-1])
        parameters['b'+str(i)] = np.zeros((layers_dim[i],1))
        #print(str(i)+"+"+str(parameters["w"+str(i)].shape)+"+"+str(parameters['b'+str(i)].shape))
    return parameters

def init_adam(parameters):
    L = len(parameters)//2
    v = {}
    s = {}

    for l in range(L):
        v["dw" + str(l + 1)] = np.zeros_like(parameters["w" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

        s["dw" + str(l + 1)] = np.zeros_like(parameters["w" + str(l + 1)])
        s["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])
    return (v,s)

def linear_forward(A,w,b):
    z = np.dot(w,A)+b
    cache = (A,w,b)
    return z,cache

def linear_activation_forward(A_prev,w,b,activation):
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, w, b)
        A,activation_cache = sigmoid(Z)
    elif activation =='relu':
        Z, linear_cache = linear_forward(A_prev, w, b)
        A ,activation_cache= relu(Z)
    cache = (linear_cache,activation_cache)
    return A,cache

def L_model_forward(X,parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1,L):
        A_prev = A
        A,cache = linear_activation_forward(A_prev,parameters['w'+str(l)],parameters['b'+str(l)],'relu')
        caches.append(cache)
    A_prev = A
    AL,cache = linear_activation_forward(A_prev,parameters['w'+str(L)],parameters['b'+str(L)],'sigmoid')
    caches.append(cache)
    return AL,caches

def compute_cost(Y,AL):
    m = Y.shape[1]
    cost =  -1/m*np.sum(np.multiply(np.log(AL),Y)+np.multiply(np.log(1-AL),(1-Y)))
    cost = np.squeeze(cost)
    return cost

# def comput_arccuracy(Y,AL):
#     Y = Y.T
#     AL = AL.T
#     errline = 0
#     for i in range(len(Y)):
#         if(AL[i]>=0.5):
#
#         if Y[i]!=AL[i]:
#             errline += 1
#     return (len(Y)-errline)/len(Y)

def linear_backward(dz,cache):
    A_prev,w,b = cache
    m = A_prev.shape[1]

    dw = np.dot(dz,A_prev.T)/m
    db = np.sum(dz,axis=1,keepdims=True)/m
    dA_prev = np.dot(w.T,dz)

    return dA_prev,dw,db

def linear_activation_backward(dA,cache,activation):
    linear_cache,activation_cache = cache
    if activation == "relu":
        dz = relu_backward(dA,activation_cache)
        dA_prev,dw,db = linear_backward(dz,linear_cache)
    elif activation == "sigmoid":
        dz = sigmoid_backward(dA,activation_cache)
        dA_prev,dw,db = linear_backward(dz,linear_cache)
    return dA_prev,dw,db

def L_model_backward(AL,Y,caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = -(np.divide(Y,AL)-np.divide(1-Y,1-AL))
    current_cache = caches[L-1]
    grads["dA"+str(L)],grads["dw"+str(L)],grads["db"+str(L)] = linear_activation_backward(dAL,current_cache,"sigmoid")
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp,dw_temp,db_temp = linear_activation_backward(grads["dA"+str(l+2)],current_cache,"relu")
        grads["dA"+str(l+1)] = dA_prev_temp
        grads["dw"+str(l+1)] = dw_temp
        grads["db"+str(l+1)] = db_temp
    return grads

def update_parameters(parameters,grads,learning_rate = 0.1):
    L = len(parameters)//2
    for l in range(L):
        parameters["w"+str(l+1)] -= learning_rate*grads["dw"+str(l+1)]
        parameters["b"+str(l+1)] -= learning_rate*grads["db"+str(l+1)]
    return parameters

def update_parameters_with_Adam(parameters,grads,v,s,t,learning_rate = 0.01,beta1 = 0.9,beta2 = 0.999,epsilon = 1e-8):
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for l in range(L):
        v["dw"+str(l+1)] = beta1*v["dw"+str(l+1)]+(1-beta1)*grads["dw"+str(l+1)]
        v["db"+str(l+1)] = beta1*v["db"+str(l+1)]+(1-beta1)*grads["db"+str(l+1)]

        v_corrected["dw"+str(l+1)] = v["dw"+str(l+1)]/(1-np.power(beta1,t))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))

        s["dw"+str(l+1)] = beta2 * s["dw"+str(l+1)]+(1-beta2)*np.square(grads["dw"+str(l+1)])
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.square(grads["db" + str(l + 1)])

        s_corrected["dw"+str(l+1)] = s["dw"+str(l+1)]/(1-np.power(beta2,t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))

        parameters["w" + str(l + 1)] = parameters["w" + str(l + 1)] - learning_rate * (v_corrected["dw" + str(l + 1)] / np.sqrt(s_corrected["dw" + str(l + 1)] + epsilon))
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * (v_corrected["db" + str(l + 1)] / np.sqrt(s_corrected["db" + str(l + 1)] + epsilon))
    return (parameters,v,s)

def dnn_model(X_total,Y_total,layers_dims,num_iterations = 3000,print_cost = True):
    np.random.seed(1)
    costs = []
    batchsize = 16
    parameters = init_parameter(layers_dims)
    v,s = init_adam(parameters)
    t = 0
    for i in range(0,num_iterations):
        X,Y = batch_data(X_total,Y_total,batchsize)
        AL,caches = L_model_forward(X,parameters)
        cost = compute_cost(Y,AL)
        # acc = comput_arccuracy(Y,AL)
        grads = L_model_backward(AL,Y,caches)
        t = t+1
        parameters,v,s = update_parameters_with_Adam(parameters,grads,v,s,t)
        if print_cost and i%100 == 0:
            print("cost  after iteration %i:%f"%(i,cost))
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens')
    plt.show()
    test_file_path = "C:\\Users\\10372\\Desktop\\Learning\\机器学习\\数据集\\考生本地用资源\\test_data.txt"
    test_x,test_y = data(test_file_path)
    AL, caches = L_model_forward(test_x, parameters)
    cost = compute_cost(test_y, AL)
    print("test cost is %f"%(cost))
    return parameters

def predict(x,y,parameters):
    m = x.shape[1]
    n = len(parameters)//2
    p = np.zeros((1,m))
    probas,caches = L_model_forward(X,parameters)
    for i in range(0,probas.shape[1]):
        if probas[0,i]>0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    print("Accuracy:"+str(np.sum((p ==Y)/m)))
    return p

def main():
    train_file_path = "C:\\Users\\10372\\Desktop\\Learning\\机器学习\\数据集\\考生本地用资源\\train_data.txt"
    x,y = data(train_file_path)
    layers_dim = (1000,100,10,1)
    parameters = dnn_model(x,y,layers_dim)

if __name__ == '__main__':
    main()
