import numpy as np
import matplotlib.pyplot as plt

def zero_pad(X,pad):
    x_paded = np.pad(X,(
        (0,0),
        (pad,pad),
        (pad,pad),
        (0,0)),
        'constant',constant_values=0)
    return x_paded

def conv_single_step(a_slice_prev,w,b):
    s = np.multiply(a_slice_prev,w)+b
    z = np.sum(s)
    return z

def conv_forward(A_prev, W, b, hparameters):

    # 获取来自上一层数据的基本信息
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # 获取权重矩阵的基本信息
    (f, f, n_C_prev, n_C) = W.shape

    # 获取超参数hparameters的值
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[0, 0, 0, c])
    assert (Z.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, W, b, hparameters)

    return (Z, cache)

def pool_forward(A_prev,hparameters,mode = "max"):
    (m, n_H_prev, n_W_prev, n_c_prev) = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]

    n_H = int((n_H_prev - f) / stride) + 1
    n_W = int((n_W_prev - f) / stride) + 1
    n_C = n_c_prev

    A = np.zeros((m,n_H,n_W,n_C))

    for i in range(m):  # 遍历样本
        for h in range(n_H):  # 在输出的垂直轴上循环
            for w in range(n_W):  # 在输出的水平轴上循环
                for c in range(n_C):  # 循环遍历输出的通道
                    # 定位当前的切片位置
                    vert_start = h * stride  # 竖向，开始的位置
                    vert_end = vert_start + f  # 竖向，结束的位置
                    horiz_start = w * stride  # 横向，开始的位置
                    horiz_end = horiz_start + f  # 横向，结束的位置
                    # 定位完毕，开始切割
                    a_slice_prev = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    # 对切片进行池化操作
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_slice_prev)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_slice_prev)

    cache = (A_prev,hparameters)
    return (A,cache)


def conv_backward(dZ, cache):
    # 获取cache的值
    (A_prev, W, b, hparameters) = cache

    # 获取A_prev的基本信息
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # 获取dZ的基本信息
    (m, n_H, n_W, n_C) = dZ.shape

    # 获取权值的基本信息
    (f, f, n_C_prev, n_C) = W.shape

    # 获取hparaeters的值
    pad = hparameters["pad"]
    stride = hparameters["stride"]

    # 初始化各个梯度的结构
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    # 前向传播中我们使用了pad，反向传播也需要使用，这是为了保证数据结构一致
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    # 现在处理数据
    for i in range(m):
        # 选择第i个扩充了的数据的样本,降了一维。
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # 定位切片位置
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f

                    # 定位完毕，开始切片
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # 切片完毕，使用上面的公式计算梯度
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
        # 设置第i个样本最终的dA_prev,即把非填充的数据取出来。
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

    # 数据处理完毕，验证数据格式是否正确
    assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return (dA_prev, dW, db)


def create_mask_from_window(x):
    mask = x == np.max(x)

    return mask


def distribute_value(dz, shape):
    # 获取矩阵的大小
    (n_H, n_W) = shape

    # 计算平均值
    average = dz / (n_H * n_W)

    # 填充入矩阵
    a = np.ones(shape) * average

    return a

def pool_backward(dA, cache, mode="max"):
    # 获取cache中的值
    (A_prev, hparaeters) = cache

    # 获取hparaeters的值
    f = hparaeters["f"]
    stride = hparaeters["stride"]

    # 获取A_prev和dA的基本信息
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (m, n_H, n_W, n_C) = dA.shape

    # 初始化输出的结构
    dA_prev = np.zeros_like(A_prev)

    # 开始处理数据
    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # 定位切片位置
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f

                    # 选择反向传播的计算方式
                    if mode == "max":
                        # 开始切片
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # 创建掩码
                        mask = create_mask_from_window(a_prev_slice)
                        # 计算dA_prev
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])

                    elif mode == "average":
                        # 获取dA的值
                        da = dA[i, h, w, c]
                        # 定义过滤器大小
                        shape = (f, f)
                        # 平均分配
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)
    # 数据处理完毕，开始验证格式
    assert (dA_prev.shape == A_prev.shape)

    return dA_prev








