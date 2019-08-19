# DeepTest modified: Automated testing of deep learning based object detection

This is a modified version of DeepTest for CNNs used for object detection. This repo obtains an overview of neuron coverage (together with evaluating performance) of the network on an input image and comparing this to multiple augmented images.


## Usage
```
python3 cnn_coverage.py path/to/data/folder
```
Notice that the data must contain imagefiles with assosiated .xml label files.
Optional arguments:
* --weights path/to/weights.h5
* --classes path/to/classes.csv
* --results path/to/results.csv

Example of usage:
```
python3 cnn_coverage.py data/data --weights ../keras-retinanet/snapshots/resnet50_csv_10_inference.h5 --results data/results.csv --classes ../keras-retinanet/images/csv/class_id.csv
```

If --weights are not specified, the default path will be ../keras-retinanet/snapshots/resnet50_coco_best_v2.1.0.h5. If --classes are not specified the default classes used for the predictions are the classes assosiated with the coco dataset.
* Important: If either --classes or --weights is specified, then the other argument follow.

The last argument, --results, expects the path to a csv-file containing the previous results of the test, i.e. this argument allows you to continue the training from previous runs. 

## Citation
If you find DeepTest useful for your research, please cite the following [paper](https://arxiv.org/pdf/1708.08559.pdf):

```
@article{tian2017deeptest,
  title={DeepTest: Automated testing of deep-neural-network-driven autonomous cars},
  author={Tian, Yuchi and Pei, Kexin and Jana, Suman and Ray, Baishakhi},
  journal={arXiv preprint arXiv:1708.08559},
  year={2017}
}
