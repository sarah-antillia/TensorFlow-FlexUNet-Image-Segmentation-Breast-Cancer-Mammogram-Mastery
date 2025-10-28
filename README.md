<h2>TensorFlow-FlexUNet-Image-Segmentation-Breast-Cancer-Mammogram-Mastery (2025/10/28)</h2>

This is the first experiment of Image Segmentation for Breast Cancer Mammogram-Mastery Singleclass,
 based on our 
TensorFlowFlexUNet (TensorFlow Flexible UNet Image Segmentation Model for Multiclass) 
and a 512x512 pixels 
<a href="https://drive.google.com/file/d/1zuwbxg-CMeqiZcTqyEjWy0BTNSBEJmFj/view?usp=sharing">
Augmented-Mammogram-Mastery-ImageMask-Dataset.zip</a>.
which was derived by us from <br><br>
<b>Breast Cancer Dataset/Original Dataset/Cancer/</b> in 
<a href="https://data.mendeley.com/datasets/fvjhtskg93/1">
<b>
Mammogram Mastery: A Robust Dataset for Breast Cancer Detection and Medical Education
</b>
</a>
<br>
<br>
As demonstrated in <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-STARE-Retinal-Vessel">
TensorFlow-FlexUNet-Image-Segmentation-STARE-Retinal-Vessel</a>, 
our Multiclass TensorFlowFlexUNet, which uses categorized masks, can also be applied to single-class image segmentation models. 
This is because it inherently treats the background as one category and your single-class mask data as a second category. 
In essence, your single-class segmentation model will operate with two categorized classes within our Multiclass UNet framework.
<br>
<br>
<hr>
<b>Acutual Image Segmentation for 512x512 pixels Cancer images</b><br>

As shown below, the inferred masks predicted by our segmentation model trained on the 
PNG dataset appear similar to the ground truth masks,
<br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test/images/barrdistorted_1002_0.3_0.3_2432.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test/masks/barrdistorted_1002_0.3_0.3_2432.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test_output/barrdistorted_1002_0.3_0.3_2432.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test/images/barrdistorted_1003_0.3_0.3_1211.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test/masks/barrdistorted_1003_0.3_0.3_1211.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test_output/barrdistorted_1003_0.3_0.3_1211.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test/images/barrdistorted_1003_0.3_0.3_2654.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test/masks/barrdistorted_1003_0.3_0.3_2654.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test_output/barrdistorted_1003_0.3_0.3_2654.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1. Dataset Citation</h3>
The dataset used was obtained from the web-site:<br><br>
<b>Breast Cancer Dataset/Original Dataset/Cancer</b> in 

<a href="https://data.mendeley.com/datasets/fvjhtskg93/1">
<b>
Mammogram Mastery: A Robust Dataset for Breast Cancer Detection and Medical Education
</b>
</a>
<br><br>
<b>Contributors</b><br>:
Karzan Barzan Aqdar,Peshraw Ahmed Abdalla,Rawand Kawa Mustafa,Zhiyar Hamid Abdulqadir,Abdalbasit Mohammed Qadir<br>
,Alla Abdulqader Shali,Nariman Muhamad Aziz
<br>
<br>
<b>Description</b><br>
This dataset presents a comprehensive data comprising breast cancer images collected from patients, encompassing two distinct sets:
 one from individuals diagnosed with breast cancer and another from those without the condition. <br>
 The dataset is meticulously curated, vetted, and classified by specialist clinicians, ensuring its reliability and accuracy
  for research and educational purposes. <br>
  Hailing from Iraq-Sulaymaniyah, the dataset offers a unique perspective on breast cancer prevalence and characteristics 
  in the region. <br>
  With 745 original images and 9,685 augmented images, this dataset provides a rich resource for training and evaluating 
  deep learning algorithms aimed at breast cancer detection. <br>
  The dataset's inclusion of augmented X-rays offers enhanced versatility for algorithm development and educational initiatives. 
  This dataset holds immense potential for advancing medical research, aiding in the development of innovative diagnostic tools, 
  and fostering educational opportunities for medical students interested in breast cancer detection and diagnosis.<br><br>
<b>Licence:</b><br>
<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0 </a>
<br>

<h3>
<a id="2">
2 Mammogram-Mastery ImageMask Dataset
</a>
</h3>
<h4>2.1 Download ImageMask Dataset</h4>
 If you would like to train this Mammogram-Mastery Segmentation model by yourself,
 please download the dataset from the google drive 
<a href="https://drive.google.com/file/d/1zuwbxg-CMeqiZcTqyEjWy0BTNSBEJmFj/view?usp=sharing">
Augmented-Mammogram-Mastery-ImageMask-Dataset.zip</a>.
<br>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Mammogram-Mastery
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Mammogram-Mastery Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Mammogram-Mastery/Mammogram-Mastery_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br>
<br>
<h4>2.2 ImageMask Dataset Derivation</h4>

<b>(1) Cropping Images </b><br>
 
 We generated a 512x512 pixels cropped PNG dataset from 1920x1080 pixels JPG images in <b>Cancer</b> folder.<br>
<pre>
./Breast Cancer Dataset
└─Original Dataset 
    ├─Cancer
    └─Non-Cancer
</pre>

<b>(2) Generation annotation dataset</b><br>
Since the masks (annotations) data were not provided for the original cancer images of the <b>Breast Cancer Dataset</b>, 
we generated our own PNG mask files correspong to the cropped PNG images by using 
a pretrained model <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Breast-Cancer-INbreast">
TensorFlow-FlexUNet-Image-Segmentation-Breast-Cancer-INbreast</a> without a manual annotation by human experts</a>,
because the INbreast cancer images appeared similar to the images in <b>Cancer</b> subset of <b>Mammogram Mastery</b>.
<br> <br>
<b>(3) Offline Dataset Augmentation</b><br>
To address the limited size of the cropped PNG images and their corresponding masks, 
we applied our offline augmentation tools 
<a href="https://github.com/sarah-antillia/ImageMask-Dataset-Offline-Augmentation-Tool"> 
ImageMask-Dataset-Offline-Augmentation-Tool</a> and 
<a href="https://github.com/sarah-antillia/Barrel-Image-Distortion-Tool">Barrel-Image-Distortion-Tool</a>  
to the cropped dataset. 
<br>
<br>
<h4>2.3 Train Images and Masks Sample</h4>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained Mammogram-Mastery TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Mammogram-Mastery/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Mammogram-Mastery and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 2

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.04
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for Mammogram-Mastery 1+3 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
; 1+1 classes
; RGB colors   cancer:white     
rgb_map = {(0,0,0):0,(255,255,255):1 }


</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInfereuncer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 17,18,19)</b><br>
<img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 35,36,37)</b><br>
<img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>
In this experiment, the training process was stopped at epoch 37 by EearlyStopping callback.<br><br>
<img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/asset/train_console_output_at_epoch37.png" width="920" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/Mammogram-Mastery/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Mammogram-Mastery/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Mammogram-Mastery</b> folder,<br>
and run the following bat file to evaluate TensorFlowFlexUNet model for Mammogram-Mastery.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/asset/evaluate_console_output_at_epoch37.png" width="920" height="auto">
<br><br>

<a href="./projects/TensorFlowFlexUNet/Mammogram-Mastery/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Mammogram-Mastery/test was very low and dice_coef_multiclass 
very high as shown below.
<br>
<pre>
categorical_crossentropy,0.0054
dice_coef_multiclass,0.9976
</pre>
<br>

<h3>
5 Inference
</h3>
Please move <b>./projects/TensorFlowFlexUNet/Mammogram-Mastery</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for Mammogram-Mastery.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of 512x512 pixels Cancer images</b><br>
<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test/images/barrdistorted_1002_0.3_0.3_3629.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test/masks/barrdistorted_1002_0.3_0.3_3629.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test_output/barrdistorted_1002_0.3_0.3_3629.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test/images/barrdistorted_1003_0.3_0.3_2654.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test/masks/barrdistorted_1003_0.3_0.3_2654.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test_output/barrdistorted_1003_0.3_0.3_2654.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test/images/barrdistorted_1004_0.3_0.3_1947.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test/masks/barrdistorted_1004_0.3_0.3_1947.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test_output/barrdistorted_1004_0.3_0.3_1947.png" width="320" height="auto"></td>
</tr>



<tr>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test/images/barrdistorted_1004_0.3_0.3_1497.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test/masks/barrdistorted_1004_0.3_0.3_1497.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test_output/barrdistorted_1004_0.3_0.3_1497.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test/images/barrdistorted_1005_0.3_0.3_4322.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test/masks/barrdistorted_1005_0.3_0.3_4322.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test_output/barrdistorted_1005_0.3_0.3_4322.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test/images/deformed_alpha_1300_sigmoid_6_4656.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test/masks/deformed_alpha_1300_sigmoid_6_4656.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram-Mastery/mini_test_output/deformed_alpha_1300_sigmoid_6_4656.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. INbreast: toward a full-field digital mammographic database </b><br>
Inês C Moreira, Igor Amaral, Inês Domingues, António Cardoso, Maria João Cardoso, Jaime S Cardoso<br>
<a href="https://pubmed.ncbi.nlm.nih.gov/22078258/">https://pubmed.ncbi.nlm.nih.gov/22078258/</a>
<br><br>
<b>2. Breast-Cancer-Segmentation-Datasets</b>
<br>
Curated collection of datasets for breast cancer segmentation
</a>
<br>
pablogiaccaglia<br>
<a href="https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/blob/master/README.md">
Breast-Cancer-Segmentation-Datasets
</a>
<br>
<br>
<b>3. TensorFlow-FlexUNet-Image-Segmentation-Breast-Cancer-INbreast</b><br>
Toshiyuki Arai antillia.com<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Breast-Cancer-INbreast">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Breast-Cancer-INbreast
</a>
<br>
<br>
<b>4. TensorFlow-FlexUNet-Image-Segmentation-MIAS-Mammogram</b><br>
Toshiyuki Arai antillia.com<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-MIAS-Mammogram">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-MIAS-Mammogram
</a>

