import numpy as np

def data(file_path):
    data_train = []
    with open(file_path, "r") as f:
        data = f.readlines()
        for line in data:
            numbers = line.split(",")
            numbers_float = list(map(float, numbers))
            # print(len(numbers_float))
            # print(numbers_float)
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

def sigmod(x):
    return 1/(1+np.exp(-x))

def relu(x):
    if x >= 0:
        return x
    else:
        return 0

def layers_size(X,Y,hidden):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    n_h = hidden
    return (n_x,n_h,n_y)

def init_parameter(n_x,n_y,n_h):
    w1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    w2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))

    parameters = {
        'w1':w1,
        'b1':b1,
        'w2':w2,
        'b2':b2
    }
    return parameters

def forward_propagation(X,parameters):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    Z1 = np.dot(w1,X)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(w2,A1)+b2
    A2 = sigmod(Z2)
    cache = {
        'A1':A1,
        'Z1':Z1,
        'A2':A2,
        'Z2':Z2
    }
    return A2,cache

def compute_cost(Y,A2,pamameters):
    m = Y.shape[1]
    cost =  -1/m*np.sum(np.multiply(np.log(A2),Y)+np.multiply(np.log(1-A2),(1-Y)))
    cost = np.squeeze(cost)

    return cost

# def compute_accuracy(Y,A2):
#     num = 0
#     for i in range(0,len(A2)):
#         if A2[i] <=0:
#             A2[i] = 0
#         else :
#             A2[i] =1
#     compare = Y - A2
#     compare = np.array(compare)
#     num = sum(compare == 0)
#     print(compare)
#     print(num)
#     return num/len(compare)

def backward_propagation(parameters,X,Y,cache):
    m = X.shape[1]
    w1 = parameters['w1']
    w2 = parameters['w2']
    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dw2 = 1/m * np.dot(dZ2,A1.T)
    db2 = 1/m * np.sum(dZ2,axis=1,keepdims=True)

    dZ1 = np.dot(w2.T,dZ2) * (1-np.power(A1,2))
    dw1 = 1/m * np.dot(dZ1,X.T)
    db1 = 1/m * np.sum(dZ1,axis=1,keepdims=True)

    grads = {
        'dw1':dw1,
        'db1':db1,
        'dw2':dw2,
        'db2':db2
    }
    return grads

def update_parameters(parameters,grads,learning_rate = 0.1):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    dw1 = grads['dw1']
    db1 = grads['db1']
    dw2 = grads['dw2']
    db2 = grads['db2']

    w1 -= dw1 * learning_rate
    w2 -= dw2 * learning_rate
    b1 -= db1 * learning_rate
    b2 -= db2 * learning_rate

    parameters = {
        'w1': w1,
        'b1': b1,
        'w2': w2,
        'b2': b2
    }
    return parameters

def nn_model(X_total,Y_total,hidden,num_iterations = 10000,print_cost = True):
    n_x = layers_size(X_total,Y_total,hidden)[0]
    n_h = layers_size(X_total,Y_total,hidden)[1]
    n_y = layers_size(X_total, Y_total, hidden)[2]

    parameters = init_parameter(n_x,n_y,n_h)
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w1']
    b2 = parameters['b2']

    for i in range(0,num_iterations):
        X,Y = batch_data(X_total,Y_total,128)
        A2,cache = forward_propagation(X,parameters)
        cost = compute_cost(Y,A2,parameters)
        grads = backward_propagation(parameters,X,Y,cache)
        parameters = update_parameters(parameters,grads)
        if print_cost and i% 100 ==0:
            print("Cost after iteration %i : %f"%(i,cost))
            # accuracy = compute_accuracy(Y,A2)
            # print("Accuracy after iteration %i : %f" % (i, accuracy))

    return parameters

def main():
    file_path = "C:\\Users\\10372\\Desktop\\Learning\\机器学习\\数据集\\考生本地用资源\\train_data.txt"
    x,y = data(file_path)

    parameters = nn_model(x,y,hidden=100)

if __name__ == '__main__':
    main()
