# MLC-VQA

MLC-VQA, or Machine Learning Codec Video Quality Assessment, is a tool designed to assess the subjective quality of videos compressed using machine learning based codecs. The system utilizes two feature extractors, SlowFast and VMAF, to extract relevant features from the video. These features are then concatenated and fed into a classification network, which produces a final MOS (mean opinion score) as the output. This tool serves as a useful tool for researchers and practitioners in the field of video quality assessment, as it allows for the analysis of the performance of machine learning based codecs.

# Installation
MLCVQA relies on pytorchvideo (should be cloned from https://github.com/facebookresearch/pytorchvideo.git) and below three repositories that are already added as git submodules (no need to be cloned):
<!--- To install the main dependencies, first clone the repositories: -->
* vmaf: https://github.com/Netflix/vmaf.git
* fb_slowfast: https://github.com/facebookresearch/SlowFast.git
* tridivb_slowfast_feature_extractor: https://github.com/tridivb/slowfast_feature_extractor.git
<!--- Then, follow the installation instructions for each repository.-->

Begin by creating mlcvqa conda enviroment to set up all depenedencies:
```cmd
conda create -n mlcvqa python=3.8 
conda activate mlcvqa
```

Install pytorchvideo and make the submodules avaialable:
```cmd
pip install -e <path_to_pytorchvideo_folder>/pytorchvideo
git submodule update --init --recursive
```


Install the depenedencies for tridivb_slowfast_feature_extractor, fb_slowfast and VMAF: 
<!--- sh slowfastDepsSetup.sh mlcvqa -->
```cmd
cd <path_to_MLVCQA_folder>
pip install -r requirements.txt
```

Next, install SlowFast and its further depenedencies: In fb_slowfast\setup.py change PIL to Pillow and then run:
```cmd
pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
conda install av -c conda-forge
cd fb_slowfast/
python setup.py build develop
```
Note: if hitting error when installing detecron2 with error about "./nvidia/cublas/lib/libcublas.so.11", try "pip uninstall nvidia_cublas_cu11" first (from: "https://stackoverflow.com/questions/74394695/how-does-one-fix-when-torch-cant-find-cuda-error-version-libcublaslt-so-11-no"). 

Install VMAF
```cmd
pip3 install cython numpy meson ninja
sudo apt-get install nasm
cd tools_av_models/mlvideocodec/tools/mlc_vqa_e2e/vmaf
make clean; make
pip3 install -r python/requirements.txt
```

## Configuration files

We have three configuration files, one for VMAF, one for SlowFast and one for MLC-VQA:

- The `configs/vmaf_config.yaml` file requires you to specify:
  - The model path and the python folder within `VMAF`.
  - The run_vmaf script path.
  - The video dimensions.
- The `configs/config.yaml` file requires you to specify:
  - The model checkpoint path.
  - The output folder.
- The SlowFast configuration file is modified within the `features.py` script.

Note: MLC-VQA has been tested on Python 3.8, but it should work with newer versions of Python as well.

# Usage
<!--- 
1. Clone this repo
2. `cd mlcvqa`
3. `conda create -n mlcvqa python=3.8` & `conda activate mlcvqa`
4. Clone dependencies (VMAF, SlowFast, SlowFast_feature_extractor)
5. Refer to their documentation and follow their installation process
6. `pip install -r requirements.txt`
-->

Follow the above setup, update the paths in yaml files in mlc_vqa_e2e/configs if needed and make sure the mlcvqa environment is activated. For a single pair of videos, from folder mlvideocodec/tools/mlc_vqa_e2e run:

`python main.py --ref <path_to>/ref.yuv --dis <path_to>/dis.yuv --mlcvqa_config ./configs/mlcvqa_config.yaml --slowfast_config ./tridivb_slowfast_feature_extractor/configs/SLOWFAST_8x8_R50.yaml --vmaf_config ./configs/vmaf_config.yaml`

For a list of pair of videos:

`python main.py --dataset path/to/dataset.csv --mlcvqa_config path/to/config.yaml --slowfast_config ./tridivb_slowfast_feature_extractor/configs/SLOWFAST_8x8_R50.yaml --vmaf_config ./configs/vmaf_config.yaml`

## Dataset

For the `dataset` parameter to work properly, the provided file has to have:
- No header
- Comma separated values
- Ref first, Dis second
- Ref and Dis are full paths to the YUV videos

For example:  
path/to/ref.yuv,path/to/dis.yuv

## Result

```json
{
    "sample_name": [
            "path/to/ref.yuv_path/to/dis.yuv",
        ], 
    "pred_mos": [
            8.224920272827148,
        ]
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
