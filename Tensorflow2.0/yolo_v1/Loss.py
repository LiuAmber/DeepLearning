import numpy as np
def compute_IOU(box1,box2):
    """
    :param box1: shape = [batch_size,S,S,B,info]  ====>info=x,y,w,h
    :param box2: shape = [batch_size,S,S,B,info]  ====>info=x,y,w,h
    :return: iou
    """
    box1_area = np.multiply(box1[...,2],box1[...,3])
    box2_area = np.multiply(box2[...,2],box2[...,3])

    box_intersection_xmin = np.maximum(box1[...,0],box2[...,0])
    box_intersection_ymin = np.maximum(box1[...,1],box2[...,1])
    box_intersection_xmax = np.minimum(box1[...,2]+box1[...,0],box2[...,2]+box2[...,0])
    box_intersection_ymax = np.minimum(box1[...,3]+box1[...,1],box2[...,3]+box1[...,1])

    box_intersection_w = np.maximum(0, box_intersection_xmax-box_intersection_xmin)
    box_intersection_h = np.maximum(0, box_intersection_ymax-box_intersection_ymin)
    box_intersection_area = np.multiply(box_intersection_w , box_intersection_h)

    box_union_area = box1_area+box2_area-box_intersection_area
    return box_intersection_area/box_union_area

def yolov1_loss(y_predict,y_true):
    """
    默认B =2 C = 20
    :param y_predict: shape = (batchsize,S,S,B*5+C)
    :param y_true: shape = (batchsize,S,S,B*5+C)
    :return: loss
    """
    batch_size = y_predict.shape[0]
    S = y_predict.shape[1]

    predict_confidence = np.reshape(y_predict[...,:2],(batch_size,S,S,2))
    predict_class = np.reshape(y_predict[...,10:],(batch_size,S,S,20))
    predict_bbox = np.reshape(y_predict[...,2:10],(batch_size,S,S,2,4))

    true_confidence =np.reshape( y_true[...,:1],(batch_size,S,S,1))
    #true_confidence = np.tile(true_confidence,[1,1,1,2])
    true_class = np.reshape(y_true[...,10:],(batch_size,S,S,20))
    true_bbox =np.reshape( y_true[...,2:6],(batch_size,S,S,1,4))
    true_bbox = np.tile(true_bbox,[1,1,1,2,1])

    iou_predict = compute_IOU(predict_bbox,true_bbox)
    object_mask = np.reshape(np.maximum(iou_predict[...,0],iou_predict[...,1]),(batch_size,S,S,1))
    object_mask = (iou_predict>=object_mask)
    object_mask = np.multiply(object_mask,true_confidence)
    noobject_mask = np.ones_like(object_mask,dtype = np.float)-object_mask

    lambda_coord = 5
    lambda_noobj = 0.5
    class_loss = np.sum(true_confidence*np.square((true_class-predict_class)))
    box_xy_loss = lambda_coord*np.sum(object_mask*np.add(np.square(predict_bbox[...,0]-true_bbox[...,0]),
                                            np.square(predict_bbox[...,1]-true_bbox[...,1])))
    box_wh_loss = lambda_coord*np.sum(object_mask*np.add(np.square(np.sqrt(predict_bbox[...,2])-np.sqrt(true_bbox[...,2])),
                                                         np.square(np.sqrt(predict_bbox[...,3])-np.sqrt(true_bbox[...,3]))))
    confidence_obj_loss = np.sum(object_mask*np.square(predict_confidence-iou_predict))
    confidence_noobj_loss = lambda_noobj*np.sum(noobject_mask*np.square(predict_confidence-true_confidence))
    yolo_loss = class_loss+box_wh_loss+box_xy_loss+confidence_noobj_loss+confidence_obj_loss
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

