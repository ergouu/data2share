#!/bin/python3
# -*- encoding: utf-8 -*-
'''
@IDE     :   vscode
@File    :   model_wrapper.py
@Time    :   2023/06/25 15:59:41
@Author  :   ergouu 
@Version :   1.0
@Contact :   ergouu@vip.qq.com
'''

# here put the import lib

import tensorflow as tf
import os

class ResizeWithCropOrPad(tf.keras.layers.Layer):
    def __init__(self,height,width,trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(ResizeWithCropOrPad,self).__init__(trainable, name, dtype, dynamic, **kwargs)
        self._height=height
        self._width=width

    @tf.function
    def call(self,inputs,*args, **kwargs):
        return tf.image.resize_with_crop_or_pad(inputs,self._height,self._width)

class Resize(tf.keras.layers.Layer):
    def __init__(self,height,width,method=None,trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(Resize,self).__init__(trainable, name, dtype, dynamic, **kwargs)
        self._height=height
        self._width=width
        self._method=method

    @tf.function
    def call(self,inputs,*args, **kwargs):
        # return tf.image.resize(inputs,size=[self._height,self._width],method=self._method,preserve_aspect_ratio=True)
        return tf.image.resize_with_pad(inputs,self._height,self._width,self._method)

class PreProcess(tf.keras.layers.Layer):
    def __init__(self,
                 layers, 
                 *args, **kwargs):
        super(PreProcess,self).__init__(*args, **kwargs)
        
        assert isinstance(layers,list),'layers is not LIST!'
        self._seq=[]

        supported_ops=['resize','resize_with_crop_or_pad','normalization']

        for layer in layers:
            layer=layer.split(',')
            assert layer[0] in supported_ops, layer[0]+' is not SUPPORTED OPS!'
            if layer[0] == 'resize':
                assert len(layer)>=3, 'please input ops, target height and width'
                height=int(layer[1])
                width=int(layer[2])
                method='bilinear'

                if len(layer)>3:
                    assert layer[3] in ['area','bicubic','bilinear','gaussian','lanczos3','lanczos5','mitchellcubic','nearest'],\
                            layer[3]+' not SUPPORTED!,\
                            [\'area\',\'bicubic\',\'bilinear\',\'gaussian\',\'lanczos3\',\'lanczos5\',\'mitchellcubic\',\'nearest\']'
                    method=layer[3]
                self._seq.append(Resize(height=height,width=width,method=method))

            elif layer[0] == 'resize_with_crop_or_pad':
                assert len(layer)==3, 'please input ops,target height and width!'
                height=int(layer[1])
                width=int(layer[2])
                self._seq.append(ResizeWithCropOrPad(height=height,width=width))

            elif layer[0] == 'normalization':
                assert len(layer)==3, 'please input ops, min and max!'
                min_value=int(layer[1])
                if min_value == -1:
                    offset=-1.0
                    scale=1/127.5
                else:
                    offset=0.0
                    scale=1/255.0
                self._seq.append(tf.keras.layers.Rescaling(scale=scale,offset=offset))
    
    def call(self,inputs):
        x=inputs
        for layer in self._seq:
            x=layer(x)
        return x

class ModelWrapper(tf.keras.Model):
    def __init__(self,
                 model_path,
                 preprocess_layers,
                 model_type,
                 model_input_resolution=[512,512]
                 ):
        super(ModelWrapper, self).__init__()
        # load the original model
        assert os.path.exists(model_path),model_path+" File NOT FOUND." 
        assert model_type in ['classic_seg','classic_det','classic_cls','yolov8seg'],\
                model_type+' is not in support list:classic_seg, classic_det, classic_cls, yolov8seg'

        self._flatten=tf.keras.layers.Flatten()
        self._preprocess=PreProcess(preprocess_layers)
        self._ori_model=tf.saved_model.load(model_path)
        self._model_input_resolution=model_input_resolution

        self._postprocess=self._cls_postprocess
        if model_type == 'classic_seg':
            self._postprocess=self._seg_postprocess
        elif model_type == 'classic_det':
            self._postprocess=self._det_postprocess
        elif model_type == 'yolov8seg':
            self._postprocess=self._yolov8_seg_postprocess

        self._finish1=tf.constant(10086,dtype=tf.int32,shape=[1,6])
        self._finish2=tf.constant(10086,dtype=tf.int32,shape=[1,2])
        

    @tf.function
    def call(self, inputs):
        x = self._preprocess(inputs)
        # st1=time.time()
        x=self._ori_model(x)
        # st2=time.time()
        x=self._postprocess(x)
        # tf.print('Model:',st2-st1,'Postprocess:',time.time()-st2)
        return x
    
    def _seg_postprocess(self,x):
        return x
    
    def _det_postprocess(self,x):
        return x
    
    def _cls_postprocess(self,x):
        return x
    
    def _yolov8_seg_postprocess(self,x,nms_iou_threshold=0.75,nms_score_threshold=0.25,mask_threshold=0.5):
        boxes,proto=x
        prob=boxes[0,4:8,:]
        coeff=boxes[0,8:,:]
        boxes=boxes[0,:4,:]
        proto=proto[0]
        cls=tf.math.argmax(prob,axis=0)
        scores=tf.math.reduce_max(prob,axis=0)

        boxes_t=tf.transpose(boxes,[1,0])
        boxes_t=tf.concat([boxes_t[:,:2]-boxes_t[:,2:]/2,boxes_t[:,:2]+boxes_t[:,2:]/2],axis=-1)
        boxes_t=tf.gather(boxes_t,[1,0,3,2],axis=-1)
        max_boxes=tf.shape(scores)[0]

        each_batch_boxes_idx=tf.raw_ops.NonMaxSuppressionV3(boxes=boxes_t,scores=scores,max_output_size=max_boxes,iou_threshold=nms_iou_threshold,score_threshold=nms_score_threshold)
        boxes_selected=tf.gather(boxes_t,each_batch_boxes_idx)
        each_cls_selected=tf.gather(cls,each_batch_boxes_idx)
        each_cls_selected=tf.cast(each_cls_selected,tf.float32)
        each_cls_selected=tf.expand_dims(each_cls_selected,axis=-1)

        each_socre=tf.gather(scores,each_batch_boxes_idx)
        each_socre=tf.expand_dims(each_socre,axis=-1)


        coeff=tf.gather(coeff,each_batch_boxes_idx,axis=-1,batch_dims=0)
        coeff=tf.expand_dims(coeff,axis=0)
        coeff=tf.expand_dims(coeff,axis=0)

        proto=tf.expand_dims(proto,axis=-1)

        maskes=tf.multiply(proto,coeff)

        maskes=tf.reduce_sum(maskes,axis=-2)
        maskes=tf.math.sigmoid(maskes)
        zeros_mask=tf.zeros_like(maskes)
        maskes=tf.raw_ops.Select(condition=(maskes>=mask_threshold),x=maskes,y=zeros_mask)

        h,w,_=maskes.shape
        ratio_hw=tf.cast(self._model_input_resolution[0]/h,dtype=tf.float32)
        rows=tf.range(h,dtype=tf.float32)#[352]
        rows=tf.expand_dims(rows,axis=-1)#[352,1]
        rows=tf.expand_dims(rows,axis=-1)#[352,1,1]
        cols=tf.range(w,dtype=tf.float32)#[2048]
        cols=tf.expand_dims(cols,axis=0)#[1,2048]
        cols=tf.expand_dims(cols,axis=-1)#[1,2048,1]

        x1,y1,x2,y2=tf.unstack(tf.expand_dims(tf.expand_dims(boxes_selected/ratio_hw,axis=0),axis=0),axis=-1)
        rows=tf.cast(tf.logical_and(tf.less_equal(x1,rows),tf.less_equal(rows,x2)),dtype=tf.float32)
        cols=tf.cast(tf.logical_and(tf.less_equal(y1,cols),tf.less_equal(cols,y2)),dtype=tf.float32)
        maskes=maskes*rows*cols
        maskes=tf.reduce_sum(maskes,axis=-1,keepdims=False)

        # maskes=tf.expand_dims(maskes,axis=-1)
        # fff=tf.image.draw_bounding_boxes([maskes],[tf.cast(boxes_selected,dtype=tf.int32)],[[255.0,0,0],[0,0,255.0]]).numpy()[0]

        return each_socre,each_cls_selected,boxes_selected,maskes
    
class ModelHandler:
    def __init__(self,labels) -> None:
        self.model=None
        self.labels=labels
        # self.load_network(model='/data/models/tf2/yolov8seg')
        self.load_network(model='/opt/nuclio/')
        
    def load_network(self,model):
        try:
            self.model=ModelWrapper(
            model_path=model,
            preprocess_layers=['resize,2048,1024','normalization,0,1'],
            model_type='yolov8seg',
            model_input_resolution=[2048,1024]
            )
            
            self.is_inititated = True
            
        except Exception as e:
            raise Exception(f"Cannot load model {model}: {e}")
        
    def infer(self,image,threshold):
        try:
            outputs=self.model(image)
            tf.print(outputs)
            results=[]
            if outputs:
                score,cls_id,boxes,maskes=outputs
                for i in range(score.shape[0]):
                    if score[i][0].numpy()>=threshold:
                        results.append(
                            {
                                "confidence":str(score[i][0].numpy()),
                                "label":self.labels.get(int(cls_id[i][0].numpy()),"unknown"),
                                "points":[int(j) for j in boxes[i].numpy().tolist()],
                                "type":"rectangle",
                            }
                        )
            return results
        except Exception as e:
            print(e)
            
# if __name__=='__main__':
#     model=ModelHandler({0:"0",1:"1",2:"2",3:"3",4:"4",5:"5"})
#     img=cv2.imread('/data/datasets/madian/1.png',cv2.IMREAD_COLOR)
#     img=cv2.resize(img,[1024,2048])
#     img=np.array([img],dtype=np.float32)
#     res=model.infer(img,0.5)
#     pass