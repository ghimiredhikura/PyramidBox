# Process AI_Challenger dataset using PyramidBox model 

Here we explain how to process [AI_Challenger](https://challenger.ai/dataset/keypoint) dataset using PyramidBox model.

1. [Download AI_Chellenger](https://challenger.ai/dataset/keypoint) keypoint dataset using this [script](https://github.com/bonseyes/SFD/blob/master/scripts/data/download_aichallenger.sh). Or you can download [manually](https://challenger.ai/dataset/keypoint). Keep in mind that you need to register first in order to get access to the dataset.
2. [Download PyramidBox Model](https://pan.baidu.com/s/1tSys4yfvKEJVZcxTLzNbUw). Authors provide pretrained model in [Biadu](https://pan.baidu.com/s/1tSys4yfvKEJVZcxTLzNbUw). If you have trouble downloading here we provide alternative download link in [Google Drive](https://drive.google.com/open?id=1rXwlqaWaTgsFcNaNlp9GE2Vxq4_G6Zge).

3. Now follow following steps to process AI_Challenger dataset. 

```Shell
git clone https://github.com/ghimiredhikura/PyramidBox.git
```
`face_detection_wider_format.py` is the script to process AI_Challenger dataset. Default path of dataset and model weigts file are the current dir, i.e, `./PyramidBox`. 

Once the dataset and model download are completed you can use following command to process dataset. 

```Shell
# usage example #
python2.7 face_detection_wider_format.py --data_dir=ai_challenger_keypoint_test_a_20180103 --out_dir=cropped --image_dir=keypoint_test_a_images_20180103 --json_file=keypoint_test_a_annotations_20180103.json --confidence=0.1
```

It will search face in each image and store the detection result in wider face format with detection score of each detection box. The detection results will be stored in `./annotations/` dir followed by data subset name (ex. `./annotations/ai_challenger_keypoint_test_a_20180103/keypoint_test_a_images_20180103/`).  
