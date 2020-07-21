import numpy as np
import tensorflow as tf
def compute_IOU(box1,box2):
    """
    :param box1: shape = [batch_size,S,S,B,info]  ====>info=x,y,w,h
    :param box2: shape = [batch_size,S,S,B,info]  ====>info=x,y,w,h
    :return: iou
    """
    box1_area = tf.multiply(box1[...,2],box1[...,3])
    box2_area = tf.multiply(box2[...,2],box2[...,3])

    box_intersection_xmin = tf.maximum(box1[...,0],box2[...,0])
    box_intersection_ymin = tf.maximum(box1[...,1],box2[...,1])
    box_intersection_xmax = tf.minimum(box1[...,2]+box1[...,0],box2[...,2]+box2[...,0])
    box_intersection_ymax = tf.minimum(box1[...,3]+box1[...,1],box2[...,3]+box1[...,1])

    box_intersection_w = tf.maximum(0.0, box_intersection_xmax-box_intersection_xmin)
    box_intersection_h = tf.maximum(0.0, box_intersection_ymax-box_intersection_ymin)
    box_intersection_area = tf.multiply(box_intersection_w , box_intersection_h)

    box_union_area = box1_area+box2_area-box_intersection_area
    return box_intersection_area/box_union_area

def yolov1_loss(y_predict,y_true):
    """
    默认B =2 C = 20
    :param y_predict: shape = (batchsize,S,S,B*5+C)
    :param y_true: shape = (batchsize,S,S,B*5+C)
    :return: loss
    """
    batch_size = tf.shape(y_predict)[0]
    S = tf.shape(y_predict)[1]

    predict_confidence = tf.reshape(y_predict[...,:2],(batch_size,S,S,2))
    predict_class = tf.reshape(y_predict[...,10:],(batch_size,S,S,20))
    predict_bbox = tf.reshape(y_predict[...,2:10],(batch_size,S,S,2,4))


    true_confidence =tf.reshape( y_true[...,:1],(batch_size,S,S,1))
    #true_confidence = np.tile(true_confidence,[1,1,1,2])
    true_class = tf.reshape(y_true[...,10:],(batch_size,S,S,20))
    true_bbox =tf.reshape( y_true[...,2:6],(batch_size,S,S,1,4))
    true_bbox = tf.tile(true_bbox,[1,1,1,2,1])

    iou_predict = compute_IOU(predict_bbox,true_bbox)
    object_mask = tf.reduce_max(iou_predict,3,keepdims=True)
    object_mask = tf.cast((iou_predict>=object_mask),tf.float32)*true_confidence
    noobject_mask = tf.ones_like(object_mask,dtype = tf.float32)-object_mask

    lambda_coord = 5
    lambda_noobj = 0.5
    class_delta = true_confidence*(predict_class-true_class)
    class_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(class_delta),axis = [1,2,3]),
        name = 'class_loss'
    )*lambda_coord

    coord_mask_ = tf.expand_dims(object_mask,4)
    boxes_delta = coord_mask_*(predict_bbox - true_bbox)
    boxes_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(boxes_delta),axis = [1,2,3,4]),
        name = 'boxes_loss'
    )*lambda_coord
    # box_xy_loss = lambda_coord*tf.sum(object_mask*tf.add(tf.square(predict_bbox[...,0]-true_bbox[...,0]),
    #                                         tf.square(predict_bbox[...,1]-true_bbox[...,1])))
    # box_wh_loss = lambda_coord*tf.sum(object_mask*tf.add(tf.square(tf.sqrt(predict_bbox[...,2])-tf.sqrt(true_bbox[...,2])),
    #                                                      tf.square(tf.sqrt(predict_bbox[...,3])-tf.sqrt(true_bbox[...,3]))))

    confidence_delta = object_mask*(predict_confidence-iou_predict)
    confidence_obj_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(confidence_delta),axis = [1,2,3]),
    name = 'confidence_obj_loss'
    )*lambda_coord

    confidence_noobj_delta = noobject_mask*predict_confidence
    confidence_noobj_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(confidence_noobj_delta),axis = [1,2,3]),
        name = 'confidence_noobj_loss'
    )*lambda_noobj

    yolo_loss = confidence_obj_loss+confidence_noobj_loss+class_loss+boxes_loss
    return yolo_loss




# box1 = np.ones((128,7,7,30))
# box2 = np.zeros((128,7,7,30))
#
# c = yolov1_loss(box1,box2)
#
# print(c.shape)
# rect_1 = Rectangle((box1[0],box1[1]),box1[2],box1[3],color = 'red')
# rect_2 = Rectangle((box2[0],box2[1]),box2[2],box2[3],color = 'blue')
# ax = plt.gca()
# ax.axes.add_patch(rect_1)
# ax.axes.add_patch(rect_2)
# plt.show()
