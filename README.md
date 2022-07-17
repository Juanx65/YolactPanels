# **Y**ou **O**nly **L**ook **A**t **C**oefficien**T**s
```
    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║   
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║   
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║   
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝ 
```

A simple, fully convolutional model for real-time instance segmentation. This is the code for our papers:
 - [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689)
 - [YOLACT++: Better Real-time Instance Segmentation](https://arxiv.org/abs/1912.06218)

# Installation
 - Clone this repository and enter it:
   ```Shell
   git clone https://github.com/Juanx65/YolactPanels.git
   cd YolactPanels
   ```
 - To use this repo and the ready to use env, you need to be working with cuda 13 or higher and cudnn installed.
 - Activate the environment:
      ```
       source yolact/bin/activate
      ```
# Setup Enviroment and Project
  - Setup Enviroment and download backbone used for this project
    - create the env with ` virtualenv yolact `
    - enter the env `source yolact/bin/activate`
    - install pytorch `pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113` or follow the instructions here `https://pytorch.org/`.
    - install the requirements `pip install -r requirements.txt`
    - Download Backbone: Resnet101 `resnet101_reducedfc.pth` from [here](https://drive.google.com/file/d/1tvqFPd4bJtakOlmn-uIA492g2qurRChj/view?usp=sharing).
  - Small issues with pytorch need to be fixed, if using the cu113 versions, follow this:
    - in `yolact/lib/python3.10/site-packages/torch/utils/utils/data/sampler.py`
      - in line 121 replace `generator = torch.Generator()` for `generator = torch.Generator(device='cuda')`
      - in line 128 to 133 replce 
      ```
              yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]
      ```
      for 
      ```
              yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator,device='cuda').tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator,device='cuda').tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator, device='cuda').tolist()
            yield from torch.randperm(n, generator=generator,device='cuda').tolist()[:self.num_samples % n]
      ```
    - in `yolact/lib/python3.10/site-packages/torch/storage.py`
      - in line 753 replace 
      ```
      return eval(cls.__module__)._UntypedStorage._expired(*args, **kwargs)
      ```
      for 
      ```
      return eval(cls.__module__)
      ```
       

# Evaluation

## Images
```Shell
# Display qualitative results on the specified image.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --image=my_image.png

# Process an image and save it to another file.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --image=input_image.png:output_image.png

# Process a whole folder of images.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --images=path/to/input/folder:path/to/output/folder
```
## Video
```Shell
# Display a video in real-time. "--video_multiframe" will process that many frames at once for improved performance.
# If you want, use "--display_fps" to draw the FPS directly on the frame.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=4 --video=my_video.mp4

# Display a webcam feed in real-time. If you have multiple webcams pass the index of the webcam you want instead of 0.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=4 --video=0

# Process a video and save it to another file. This uses the same pipeline as the ones above now, so it's fast!
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=4 --video=input_video.mp4:output_video.mp4
```
As you can tell, `eval.py` can do a ton of stuff. Run the `--help` command to see everything it can do.
```Shell
python eval.py --help
```


# Training
By default, we train on COCO. Make sure to download the entire dataset using the commands above.
 - To train, grab an imagenet-pretrained model and put it in `./weights`.
   - For Resnet101, download `resnet101_reducedfc.pth` from [here](https://drive.google.com/file/d/1tvqFPd4bJtakOlmn-uIA492g2qurRChj/view?usp=sharing).
   - For Resnet50, download `resnet50-19c8e357.pth` from [here](https://drive.google.com/file/d/1Jy3yCdbatgXa5YYIdTCRrSV0S9V5g1rn/view?usp=sharing).
   - For Darknet53, download `darknet53.pth` from [here](https://drive.google.com/file/d/17Y431j4sagFpSReuPNoFcj9h7azDTZFf/view?usp=sharing).
 - Run one of the training commands below.
   - Note that you can press ctrl+c while training and it will save an `*_interrupt.pth` file at the current iteration.
   - All weights are saved in the `./weights` directory by default with the file name `<config>_<epoch>_<iter>.pth`.
```Shell
# Trains using the base config with a batch size of 8 (the default).
python train.py --config=yolact_base_config

# Trains yolact_base_config with a batch_size of 5. For the 550px models, 1 batch takes up around 1.5 gigs of VRAM, so specify accordingly.
python train.py --config=yolact_base_config --batch_size=5

# Resume training yolact_base with a specific weight file and start from the iteration specified in the weight file's name.
python train.py --config=yolact_base_config --resume=weights/yolact_base_10_32100.pth --start_iter=-1

# Use the help option to see a description of all available command line arguments
python train.py --help
```

## Multi-GPU Support
YOLACT now supports multiple GPUs seamlessly during training:

 - Before running any of the scripts, run: `export CUDA_VISIBLE_DEVICES=[gpus]`
   - Where you should replace [gpus] with a comma separated list of the index of each GPU you want to use (e.g., 0,1,2,3).
   - You should still do this if only using 1 GPU.
   - You can check the indices of your GPUs with `nvidia-smi`.
 - Then, simply set the batch size to `8*num_gpus` with the training commands above. The training script will automatically scale the hyperparameters to the right values.
   - If you have memory to spare you can increase the batch size further, but keep it a multiple of the number of GPUs you're using.
   - If you want to allocate the images per GPU specific for different GPUs, you can use `--batch_alloc=[alloc]` where [alloc] is a comma seprated list containing the number of images on each GPU. This must sum to `batch_size`.

## Logging
YOLACT now logs training and validation information by default. You can disable this with `--no_log`. A guide on how to visualize these logs is coming soon, but now you can look at `LogVizualizer` in `utils/logger.py` for help.

## Pascal SBD
We also include a config for training on Pascal SBD annotations (for rapid experimentation or comparing with other methods). To train on Pascal SBD, proceed with the following steps:
 1. Download the dataset from [here](http://home.bharathh.info/pubs/codes/SBD/download.html). It's the first link in the top "Overview" section (and the file is called `benchmark.tgz`).
 2. Extract the dataset somewhere. In the dataset there should be a folder called `dataset/img`. Create the directory `./data/sbd` (where `.` is YOLACT's root) and copy `dataset/img` to `./data/sbd/img`.
 4. Download the COCO-style annotations from [here](https://drive.google.com/open?id=1ExrRSPVctHW8Nxrn0SofU1lVhK5Wn0_S).
 5. Extract the annotations into `./data/sbd/`.
 6. Now you can train using `--config=yolact_resnet50_pascal_config`. Check that config to see how to extend it to other models.

I will automate this all with a script soon, don't worry. Also, if you want the script I used to convert the annotations, I put it in `./scripts/convert_sbd.py`, but you'll have to check how it works to be able to use it because I don't actually remember at this point.

If you want to verify our results, you can download our `yolact_resnet50_pascal_config` weights from [here](https://drive.google.com/open?id=1yLVwtkRtNxyl0kxeMCtPXJsXFFyc_FHe). This model should get 72.3 mask AP_50 and 56.2 mask AP_70. Note that the "all" AP isn't the same as the "vol" AP reported in others papers for pascal (they use an averages of the thresholds from `0.1 - 0.9` in increments of `0.1` instead of what COCO uses).

## Custom Datasets
You can also train on your own dataset by following these steps:
 - Create a COCO-style Object Detection JSON annotation file for your dataset. The specification for this can be found [here](http://cocodataset.org/#format-data). Note that we don't use some fields, so the following may be omitted:
   - `info`
   - `liscense`
   - Under `image`: `license, flickr_url, coco_url, date_captured`
   - `categories` (we use our own format for categories, see below)
 - Create a definition for your dataset under `dataset_base` in `data/config.py` (see the comments in `dataset_base` for an explanation of each field):
```Python
my_custom_dataset = dataset_base.copy({
    'name': 'My Dataset',

    'train_images': 'path_to_training_images',
    'train_info':   'path_to_training_annotation',

    'valid_images': 'path_to_validation_images',
    'valid_info':   'path_to_validation_annotation',

    'has_gt': True,
    'class_names': ('my_class_id_1', 'my_class_id_2', 'my_class_id_3', ...)
})
```
 - A couple things to note:
   - Class IDs in the annotation file should start at 1 and increase sequentially on the order of `class_names`. If this isn't the case for your annotation file (like in COCO), see the field `label_map` in `dataset_base`.
   - If you do not want to create a validation split, use the same image path and annotations file for validation. By default (see `python train.py --help`), `train.py` will output validation mAP for the first 5000 images in the dataset every 2 epochs.
 - Finally, in `yolact_base_config` in the same file, change the value for `'dataset'` to `'my_custom_dataset'` or whatever you named the config object above. Then you can use any of the training commands in the previous section.

#### Creating a Custom Dataset from Scratch
See [this nice post by @Amit12690](https://github.com/dbolya/yolact/issues/70#issuecomment-504283008) for tips on how to annotate a custom dataset and prepare it for use with YOLACT.




# Citation
If you use YOLACT or this code base in your work, please cite
```
@inproceedings{yolact-iccv2019,
  author    = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
  title     = {YOLACT: {Real-time} Instance Segmentation},
  booktitle = {ICCV},
  year      = {2019},
}
```

For YOLACT++, please cite
```
@article{yolact-plus-tpami2020,
  author  = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title   = {YOLACT++: Better Real-time Instance Segmentation}, 
  year    = {2020},
}
```



# Contact
For questions about our paper or code, please contact [Daniel Bolya](mailto:dbolya@ucdavis.edu).
