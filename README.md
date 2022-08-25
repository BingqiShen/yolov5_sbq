# yolov5_sbq
make it possible to detect images in batch and modify parameters of NMS

## 1. NMS setting
in ./utils/general.py, 

```
# score_threshold: 此阈值不同于前面分类置信度阈值
# score_threshold=0.25
def my_soft_nms(bboxes, scores, iou_thresh=0.5,sigma=0.5,score_threshold=0.02):
    bboxes = bboxes.contiguous()
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    # 计算每个box的面积
    areas = (x2-x1+1)*(y2-y1+1)
    #首先对所有得分进行一次降序排列,仅此一次,以提高后续查找最大值速度. order为降序排列后的索引
    _, order = scores.sort(0, descending=True)
    # NMS后,保存留下来的边框
    keep = []
```

可以通过修改score_threshold参数来更改NMS效果

参考：https://blog.csdn.net/okgwf/article/details/122484378

score_threshold = 0意为没有NMS

## 2. detect_batch.py
```
python detect_batch.py --weights runs/train/yolo5l/weights/best.pt --source /media/thinking/thinking/
```

结果保存在 ./runs/detect/exp/中

参考：https://zhuanlan.zhihu.com/p/543657831
