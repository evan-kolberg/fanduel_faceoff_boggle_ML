# How to run this

## Step 1: Prerequisites

cuda 12.1

install pytorch with cuda support

install some caffeine into your body

prepare for headaches and errors to no avail






```
NVIDIA GeForce GTX 1660 Ti
Ultralytics YOLOv8.2.32  Python-3.11.9 torch-2.3.1 CUDA:0 (NVIDIA GeForce GTX 1660 Ti, 6144MiB)
[34m[1mengine\trainer: [0mtask=detect, mode=train, model=yolov8n.pt, data=objdetect.yaml, epochs=100, time=None, patience=100, batch=4, imgsz=1024, save=True, save_period=-1, cache=False, device=None, workers=8, project=None, name=boggle-model-8n, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=d:\programming\fanduel-boggle-cheat\runs\detect\boggle-model-8n
Overriding model.yaml nc=80 with nc=31

                   from  n    params  module                                       arguments                     
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]             
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]             
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]           
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]                  
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]                 
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
 22        [15, 18, 21]  1    757357  ultralytics.nn.modules.head.Detect           [31, [64, 128, 256]]          
Model summary: 225 layers, 3016893 parameters, 3016877 gradients, 8.2 GFLOPs

Transferred 319/355 items from pretrained weights
Freezing layer 'model.22.dfl.conv.weight'
[34m[1mAMP: [0mrunning Automatic Mixed Precision (AMP) checks with YOLOv8n...
[34m[1mAMP: [0mchecks passed 
[34m[1mtrain: [0mScanning D:\programming\fanduel-boggle-cheat\dataset\train\labels... 80 images, 0 backgrounds, 0 corrupt: 100%|██████████| 80/80 [00:01<00:00, 70.73it/s][34m[1mtrain: [0mNew cache created: D:\programming\fanduel-boggle-cheat\dataset\train\labels.cache

[34m[1mval: [0mScanning D:\programming\fanduel-boggle-cheat\dataset\val\labels... 20 images, 0 backgrounds, 0 corrupt: 100%|██████████| 20/20 [00:00<00:00, 40.41it/s][34m[1mval: [0mNew cache created: D:\programming\fanduel-boggle-cheat\dataset\val\labels.cache

Plotting labels to d:\programming\fanduel-boggle-cheat\runs\detect\boggle-model-8n\labels.jpg... 
[34m[1moptimizer:[0m 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
[34m[1moptimizer:[0m AdamW(lr=0.000286, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)
Image sizes 1024 train, 1024 val
Using 8 dataloader workers
Logging results to [1md:\programming\fanduel-boggle-cheat\runs\detect\boggle-model-8n[0m
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/100      2.67G        nan        nan        nan        140       1024: 100%|██████████| 20/20 [00:13<00:00,  1.47it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.28it/s]
                   all         20        355          0          0          0          0

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/100       2.7G        nan        nan        nan        111       1024: 100%|██████████| 20/20 [00:08<00:00,  2.30it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  4.06it/s]                   all         20        355          0          0          0          0


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/100      2.67G        nan        nan        nan        111       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  4.04it/s]                   all         20        355          0          0          0          0


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/100      2.64G        nan        nan        nan         92       1024: 100%|██████████| 20/20 [00:08<00:00,  2.31it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  4.06it/s]                   all         20        355          0          0          0          0


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/100      2.67G        nan        nan        nan        123       1024: 100%|██████████| 20/20 [00:08<00:00,  2.30it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.86it/s]                   all         20        355          0          0          0          0


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/100      2.77G        nan        nan        nan        142       1024: 100%|██████████| 20/20 [00:08<00:00,  2.28it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.81it/s]                   all         20        355          0          0          0          0


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/100      2.56G        nan        nan        nan        125       1024: 100%|██████████| 20/20 [00:08<00:00,  2.29it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.67it/s]                   all         20        355          0          0          0          0


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/100      2.63G        nan        nan        nan         52       1024: 100%|██████████| 20/20 [00:08<00:00,  2.30it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.85it/s]                   all         20        355          0          0          0          0


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/100       2.6G        nan        nan        nan        105       1024: 100%|██████████| 20/20 [00:08<00:00,  2.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  4.06it/s]                   all         20        355          0          0          0          0


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/100      2.83G        nan        nan        nan        127       1024: 100%|██████████| 20/20 [00:08<00:00,  2.27it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.87it/s]                   all         20        355    0.00443    0.00472    0.00246     0.0014


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/100      2.62G        nan        nan        nan        123       1024: 100%|██████████| 20/20 [00:08<00:00,  2.29it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  4.04it/s]                   all         20        355    0.00267    0.00322    0.00155    0.00107


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/100      2.76G        nan        nan        nan        207       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  4.01it/s]                   all         20        355     0.0015    0.00172   0.000875   0.000525


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/100      2.67G        nan        nan        nan         52       1024: 100%|██████████| 20/20 [00:08<00:00,  2.28it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  4.04it/s]
                   all         20        355    0.00241    0.00322    0.00145   0.000727

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/100      2.74G        nan        nan        nan         81       1024: 100%|██████████| 20/20 [00:08<00:00,  2.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.94it/s]
                   all         20        355    0.00603    0.00603    0.00382    0.00167

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/100      2.59G        nan        nan        nan         74       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.61it/s]
                   all         20        355    0.00235    0.00322    0.00136   0.000572

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/100      2.76G        nan        nan        nan         70       1024: 100%|██████████| 20/20 [00:08<00:00,  2.29it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.83it/s]                   all         20        355    0.00214    0.00322    0.00119   0.000292


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/100       2.7G        nan        nan        nan        118       1024: 100%|██████████| 20/20 [00:08<00:00,  2.27it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  4.00it/s]                   all         20        355    0.00215    0.00322    0.00128   0.000358


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/100      2.72G        nan        nan        nan         85       1024: 100%|██████████| 20/20 [00:08<00:00,  2.28it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.96it/s]
                   all         20        355    0.00157    0.00172    0.00127   0.000381

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/100      2.73G        nan        nan        nan        163       1024: 100%|██████████| 20/20 [00:08<00:00,  2.27it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.93it/s]
                   all         20        355    0.00181    0.00172    0.00139   0.000417

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/100      2.84G        nan        nan        nan        117       1024: 100%|██████████| 20/20 [00:08<00:00,  2.27it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  4.02it/s]                   all         20        355    0.00164    0.00172    0.00105   0.000314


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/100      2.63G        nan        nan        nan        110       1024: 100%|██████████| 20/20 [00:08<00:00,  2.28it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.95it/s]                   all         20        355    0.00144    0.00172    0.00147    0.00044


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/100      2.77G        nan        nan        nan         49       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  4.01it/s]                   all         20        355    0.00164    0.00172    0.00156   0.000313


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/100      2.63G        nan        nan        nan         64       1024: 100%|██████████| 20/20 [00:08<00:00,  2.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.83it/s]                   all         20        355    0.00731    0.00753    0.00458    0.00205


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/100      2.72G        nan        nan        nan         79       1024: 100%|██████████| 20/20 [00:08<00:00,  2.27it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.93it/s]
                   all         20        355    0.00223    0.00322    0.00149   0.000481

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/100      2.68G        nan        nan        nan         63       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.66it/s]                   all         20        355    0.00783    0.00753    0.00456    0.00203


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/100      2.66G        nan        nan        nan        123       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  4.03it/s]                   all         20        355    0.00778    0.00753    0.00443    0.00192


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/100      2.69G        nan        nan        nan        126       1024: 100%|██████████| 20/20 [00:08<00:00,  2.24it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.72it/s]
                   all         20        355    0.00274    0.00472    0.00162   0.000514

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/100      2.69G        nan        nan        nan         76       1024: 100%|██████████| 20/20 [00:08<00:00,  2.24it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.95it/s]
                   all         20        355     0.0015    0.00172    0.00103   0.000206

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/100      2.63G        nan        nan        nan        202       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  4.02it/s]
                   all         20        355    0.00157    0.00172    0.00101   0.000202

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/100      2.67G        nan        nan        nan         86       1024: 100%|██████████| 20/20 [00:08<00:00,  2.23it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.99it/s]                   all         20        355    0.00144    0.00172      0.001     0.0003


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/100      2.71G        nan        nan        nan        120       1024: 100%|██████████| 20/20 [00:08<00:00,  2.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.96it/s]                   all         20        355    0.00133    0.00172    0.00141   0.000565


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/100      2.56G        nan        nan        nan        138       1024: 100%|██████████| 20/20 [00:08<00:00,  2.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.87it/s]
                   all         20        355    0.00164    0.00172    0.00156   0.000626

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/100      2.59G        nan        nan        nan         86       1024: 100%|██████████| 20/20 [00:08<00:00,  2.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  4.02it/s]                   all         20        355    0.00657    0.00603    0.00405    0.00167


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/100      2.79G        nan        nan        nan         73       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.89it/s]
                   all         20        355    0.00569    0.00603    0.00355    0.00146

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/100      2.76G        nan        nan        nan         64       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.69it/s]                   all         20        355     0.0015    0.00172    0.00124   0.000247


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/100      2.67G        nan        nan        nan        101       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.55it/s]                   all         20        355    0.00725    0.00603    0.00429    0.00187


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/100      2.74G        nan        nan        nan         75       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  4.01it/s]                   all         20        355    0.00643    0.00603    0.00411    0.00179


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/100      2.93G        nan        nan        nan        124       1024: 100%|██████████| 20/20 [00:08<00:00,  2.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.54it/s]
                   all         20        355    0.00383    0.00431    0.00254    0.00127

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/100      2.68G        nan        nan        nan         90       1024: 100%|██████████| 20/20 [00:08<00:00,  2.24it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.98it/s]
                   all         20        355    0.00517    0.00603    0.00317    0.00121

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/100      2.68G        nan        nan        nan         84       1024: 100%|██████████| 20/20 [00:08<00:00,  2.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.98it/s]
                   all         20        355    0.00588    0.00603     0.0035    0.00139

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     41/100      2.69G        nan        nan        nan         94       1024: 100%|██████████| 20/20 [00:08<00:00,  2.24it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.96it/s]
                   all         20        355    0.00645    0.00753    0.00448    0.00178

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     42/100      2.65G        nan        nan        nan        142       1024: 100%|██████████| 20/20 [00:08<00:00,  2.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.95it/s]
                   all         20        355    0.00533    0.00603     0.0034    0.00136

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     43/100      2.59G        nan        nan        nan         54       1024: 100%|██████████| 20/20 [00:08<00:00,  2.22it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.91it/s]
                   all         20        355    0.00383    0.00431    0.00239     0.0012

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     44/100       2.8G        nan        nan        nan        100       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  4.02it/s]
                   all         20        355    0.00345    0.00431    0.00212    0.00106

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     45/100      2.86G        nan        nan        nan        110       1024: 100%|██████████| 20/20 [00:08<00:00,  2.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.65it/s]                   all         20        355    0.00588    0.00603    0.00364    0.00146


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     46/100      2.97G        nan        nan        nan        112       1024: 100%|██████████| 20/20 [00:08<00:00,  2.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.91it/s]                   all         20        355    0.00657    0.00603    0.00419    0.00174


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     47/100      2.59G        nan        nan        nan        137       1024: 100%|██████████| 20/20 [00:08<00:00,  2.27it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.71it/s]                   all         20        355    0.00591    0.00753    0.00335    0.00121


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     48/100      2.58G        nan        nan        nan         92       1024: 100%|██████████| 20/20 [00:08<00:00,  2.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.76it/s]                   all         20        355    0.00202    0.00322     0.0012   0.000411


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     49/100      2.86G        nan        nan        nan        135       1024: 100%|██████████| 20/20 [00:08<00:00,  2.27it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.95it/s]                   all         20        355    0.00705    0.00753      0.004    0.00178


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     50/100      2.65G        nan        nan        nan         69       1024: 100%|██████████| 20/20 [00:08<00:00,  2.24it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.95it/s]
                   all         20        355     0.0071    0.00903    0.00408    0.00185

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     51/100      2.73G        nan        nan        nan         93       1024: 100%|██████████| 20/20 [00:08<00:00,  2.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.98it/s]
                   all         20        355    0.00213    0.00322    0.00125   0.000385

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     52/100      2.72G        nan        nan        nan         86       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.90it/s]
                   all         20        355    0.00739    0.00603    0.00433    0.00184

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     53/100      2.65G        nan        nan        nan         75       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.99it/s]                   all         20        355    0.00144    0.00172    0.00121   0.000362


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     54/100      2.59G        nan        nan        nan         89       1024: 100%|██████████| 20/20 [00:09<00:00,  2.20it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.94it/s]
                   all         20        355     0.0015    0.00172   0.000941   0.000188

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     55/100      2.85G        nan        nan        nan        113       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  4.02it/s]                   all         20        355    0.00444    0.00603    0.00296    0.00135


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     56/100       2.6G        nan        nan        nan        118       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.55it/s]                   all         20        355    0.00409    0.00603    0.00272    0.00104


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     57/100      2.73G        nan        nan        nan        163       1024: 100%|██████████| 20/20 [00:08<00:00,  2.24it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.81it/s]                   all         20        355    0.00674    0.00603    0.00413    0.00169


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     58/100      2.63G        nan        nan        nan        124       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.78it/s]                   all         20        355    0.00657    0.00603    0.00397    0.00166


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     59/100      2.71G        nan        nan        nan        112       1024: 100%|██████████| 20/20 [00:08<00:00,  2.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  4.01it/s]                   all         20        355    0.00203    0.00322    0.00127   0.000365


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     60/100       2.7G        nan        nan        nan        183       1024: 100%|██████████| 20/20 [00:08<00:00,  2.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.96it/s]                   all         20        355    0.00199    0.00322    0.00116    0.00044


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/100      2.85G        nan        nan        nan         76       1024: 100%|██████████| 20/20 [00:08<00:00,  2.23it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.97it/s]                   all         20        355     0.0015    0.00172   0.000892   8.92e-05


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/100      2.73G        nan        nan        nan        137       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  4.01it/s]                   all         20        355     0.0015    0.00172   0.000892   0.000268


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/100      2.78G        nan        nan        nan         98       1024: 100%|██████████| 20/20 [00:08<00:00,  2.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.99it/s]                   all         20        355    0.00123    0.00172   0.000901    0.00027


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/100      2.83G        nan        nan        nan         80       1024: 100%|██████████| 20/20 [00:08<00:00,  2.27it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  4.01it/s]                   all         20        355    0.00144    0.00172      0.001     0.0002


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/100      2.62G        nan        nan        nan         96       1024: 100%|██████████| 20/20 [00:08<00:00,  2.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.49it/s]                   all         20        355    0.00133    0.00172   0.000947   0.000379


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/100      2.95G        nan        nan        nan        126       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  4.01it/s]
                   all         20        355    0.00657    0.00603    0.00395    0.00174

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/100      2.59G        nan        nan        nan         98       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.82it/s]                   all         20        355    0.00706    0.00753    0.00424    0.00195


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     68/100      2.75G        nan        nan        nan        154       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.84it/s]
                   all         20        355    0.00716    0.00753    0.00426    0.00212

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     69/100      2.56G        nan        nan        nan        102       1024: 100%|██████████| 20/20 [00:08<00:00,  2.28it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.93it/s]
                   all         20        355     0.0071    0.00753    0.00416    0.00191

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     70/100      2.73G        nan        nan        nan         99       1024: 100%|██████████| 20/20 [00:08<00:00,  2.23it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.86it/s]                   all         20        355    0.00233    0.00322    0.00142   0.000494


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     71/100       2.8G        nan        nan        nan         72       1024: 100%|██████████| 20/20 [00:08<00:00,  2.24it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  4.02it/s]                   all         20        355    0.00222    0.00322    0.00148   0.000547


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     72/100       2.7G        nan        nan        nan        111       1024: 100%|██████████| 20/20 [00:08<00:00,  2.23it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.95it/s]                   all         20        355    0.00226    0.00322    0.00133   0.000298


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     73/100      2.65G        nan        nan        nan         99       1024: 100%|██████████| 20/20 [00:08<00:00,  2.23it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.96it/s]
                   all         20        355     0.0071    0.00753    0.00414    0.00173

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     74/100      2.83G        nan        nan        nan        174       1024: 100%|██████████| 20/20 [00:09<00:00,  2.21it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.64it/s]                   all         20        355    0.00654    0.00753    0.00388     0.0017


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     75/100      2.62G        nan        nan        nan        127       1024: 100%|██████████| 20/20 [00:08<00:00,  2.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.99it/s]
                   all         20        355    0.00768    0.00753    0.00441    0.00192

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     76/100      2.78G        nan        nan        nan        109       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.40it/s]                   all         20        355    0.00787    0.00753    0.00454    0.00211


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     77/100      2.78G        nan        nan        nan        138       1024: 100%|██████████| 20/20 [00:08<00:00,  2.24it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  4.03it/s]                   all         20        355    0.00666    0.00753    0.00405    0.00175


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     78/100      2.71G        nan        nan        nan        111       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.76it/s]                   all         20        355    0.00613    0.00603    0.00376     0.0015


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     79/100      2.64G        nan        nan        nan        125       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.96it/s]                   all         20        355    0.00675    0.00753    0.00434    0.00192


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     80/100       2.9G        nan        nan        nan         71       1024: 100%|██████████| 20/20 [00:08<00:00,  2.27it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.98it/s]
                   all         20        355    0.00747    0.00603     0.0044    0.00179

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     81/100      2.78G        nan        nan        nan         87       1024: 100%|██████████| 20/20 [00:08<00:00,  2.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.96it/s]
                   all         20        355    0.00138    0.00172    0.00105    0.00021

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     82/100      2.72G        nan        nan        nan        126       1024: 100%|██████████| 20/20 [00:08<00:00,  2.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.89it/s]                   all         20        355     0.0015    0.00172    0.00103   0.000103


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     83/100      2.69G        nan        nan        nan        188       1024: 100%|██████████| 20/20 [00:08<00:00,  2.24it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.95it/s]
                   all         20        355    0.00172    0.00172    0.00114   0.000228

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     84/100      2.86G        nan        nan        nan        100       1024: 100%|██████████| 20/20 [00:08<00:00,  2.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.94it/s]                   all         20        355    0.00204    0.00322    0.00115   0.000491


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     85/100      2.72G        nan        nan        nan        118       1024: 100%|██████████| 20/20 [00:08<00:00,  2.24it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.98it/s]                   all         20        355    0.00138    0.00172    0.00105    0.00021


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     86/100      2.85G        nan        nan        nan         90       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.78it/s]                   all         20        355    0.00649    0.00603    0.00383    0.00149


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     87/100       2.8G        nan        nan        nan        179       1024: 100%|██████████| 20/20 [00:08<00:00,  2.23it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.85it/s]
                   all         20        355    0.00172    0.00172     0.0016   0.000321

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     88/100      2.73G        nan        nan        nan         86       1024: 100%|██████████| 20/20 [00:08<00:00,  2.23it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.88it/s]                   all         20        355    0.00713    0.00603    0.00405     0.0017


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     89/100      2.69G        nan        nan        nan        106       1024: 100%|██████████| 20/20 [00:08<00:00,  2.24it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.74it/s]                   all         20        355    0.00775    0.00753     0.0045    0.00201


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     90/100      2.76G        nan        nan        nan        106       1024: 100%|██████████| 20/20 [00:09<00:00,  2.22it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.85it/s]
                   all         20        355    0.00236    0.00322    0.00129   0.000446
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     91/100      2.57G        nan        nan        nan         74       1024: 100%|██████████| 20/20 [00:15<00:00,  1.28it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.89it/s]                   all         20        355     0.0128    0.00935    0.00816     0.0039


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     92/100      2.57G        nan        nan        nan         78       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.92it/s]
                   all         20        355    0.00748    0.00504    0.00507    0.00266

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     93/100      2.57G        nan        nan        nan         72       1024: 100%|██████████| 20/20 [00:08<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.96it/s]
                   all         20        355    0.00616    0.00481    0.00512    0.00371

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     94/100      2.56G        nan        nan        nan         66       1024: 100%|██████████| 20/20 [00:08<00:00,  2.24it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.86it/s]
                   all         20        355     0.0137    0.00928    0.00934    0.00729

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     95/100      2.57G        nan        nan        nan         69       1024: 100%|██████████| 20/20 [00:08<00:00,  2.28it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.91it/s]
                   all         20        355     0.0023    0.00565     0.0028    0.00229

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     96/100      2.57G        nan        nan        nan         69       1024: 100%|██████████| 20/20 [00:08<00:00,  2.27it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.61it/s]
                   all         20        355     0.0121    0.00928    0.00839    0.00591

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     97/100      2.57G        nan        nan        nan         69       1024: 100%|██████████| 20/20 [00:08<00:00,  2.29it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.97it/s]                   all         20        355     0.0109    0.00928    0.00923    0.00694


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     98/100      2.57G        nan        nan        nan         74       1024: 100%|██████████| 20/20 [00:08<00:00,  2.28it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.65it/s]
                   all         20        355    0.00502    0.00481    0.00388    0.00364

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     99/100      2.57G        nan        nan        nan         70       1024: 100%|██████████| 20/20 [00:08<00:00,  2.28it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.68it/s]                   all         20        355    0.00477    0.00747    0.00345    0.00314


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    100/100      2.57G        nan        nan        nan         82       1024: 100%|██████████| 20/20 [00:08<00:00,  2.27it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.94it/s]
                   all         20        355     0.0046    0.00481    0.00573    0.00537

100 epochs completed in 0.286 hours.
Optimizer stripped from d:\programming\fanduel-boggle-cheat\runs\detect\boggle-model-8n\weights\last.pt, 6.3MB
Optimizer stripped from d:\programming\fanduel-boggle-cheat\runs\detect\boggle-model-8n\weights\best.pt, 6.3MB

Validating d:\programming\fanduel-boggle-cheat\runs\detect\boggle-model-8n\weights\best.pt...
Ultralytics YOLOv8.2.32  Python-3.11.9 torch-2.3.1 CUDA:0 (NVIDIA GeForce GTX 1660 Ti, 6144MiB)
Model summary (fused): 168 layers, 3011693 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:00<00:00,  3.60it/s]
                   all         20        355     0.0108    0.00747     0.0087    0.00765
                     A         14         23          0          0          0          0
                     B          4          4          0          0          0          0
                     C          2          2          0          0          0          0
                     D         11         12          0          0          0          0
                     E         20         52          0          0          0          0
                     F          6          6          0          0          0          0
                     G          8          8          0          0          0          0
                     H         15         20          0          0          0          0
                     I         16         23          0          0          0          0
                     J          2          2          0          0          0          0
                     K          5          5          0          0          0          0
                     L         10         12          0          0          0          0
                     M          6          6          0          0          0          0
                     N         14         21          0          0          0          0
                     O         16         23     0.0333      0.087     0.0622     0.0594
                     P          5          5          0          0          0          0
                     R         12         13     0.0303     0.0769     0.0173    0.00692
                     S         13         19       0.25     0.0526      0.173      0.156
                     T         16         26          0          0          0          0
                     U          8          9          0          0          0          0
                     V          8          9          0          0          0          0
                     W          8          8          0          0          0          0
                     Y          7          8          0          0          0          0
                     Z          4          4          0          0          0          0
                    DL          2          6          0          0          0          0
                    DW          2          2          0          0          0          0
                    TL          5         15          0          0          0          0
                    TW          5          5          0          0          0          0
                SUBMIT          7          7          0          0          0          0
Speed: 1.4ms preprocess, 31.7ms inference, 0.1ms loss, 2.4ms postprocess per image
Results saved to [1md:\programming\fanduel-boggle-cheat\runs\detect\boggle-model-8n[0m

```