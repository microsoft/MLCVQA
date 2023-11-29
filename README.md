# MLCVQA

MLCVQA, or Machine Learning Codec Video Quality Assessment, is a tool designed to assess the subjective quality of videos compressed using machine learning based codecs. The system utilizes two feature extractors, SlowFast and VMAF, to extract relevant features from the video. These features are then concatenated and fed into a classification network, which produces a final MOS (mean opinion score) as the output. This tool serves as a useful tool for researchers and practitioners in the field of video quality assessment, as it allows for the analysis of the performance of machine learning based codecs.

MLCVQA is a model that is trained and tested on sequences with 300 frames and a resolution of 1920x1080. The current pipeline preprocesses the data to match these dimensions and length if needed. However, it is important to note that the resulting scores may not be as accurate if the frame dimension and sequence length are very different from what the model is trained on.

# Installation and Usage with Docker

## Building
Run the following command in `docker` folder:
```
docker build . -t <image_name:tag> 
```

## Usage
For a single pair of videos, run:

`docker run --rm --gpus all --shm-size 2Gb -v <path/to/repo>:/app -v <path/to/dataset>:/mnt -v <path/to/outputs>:/app/outputs <image_name:tag> python main.py --ref <path_to>/ref.yuv --dis <path_to>/dis.yuv --preprocess`

For a list of pair of videos, run

`docker run --rm --gpus all --shm-size 2Gb -v <path/to/repo>:/app -v <path/to/dataset>:/mnt -v <path/to/outputs>:/app/outputs <image_name:tag> python main.py --dataset <path_to>/dataset.csv --preprocess`

note that --dataset, --ref, and --dis must be container local paths.

## Debugging MLCVQA

Example configuration files for VSCode are in `docker/vscode` folder:
- [tasks.json](vscode/tasks.json)
- [launch.json](vscode/launch.json)

# Dataset

For the `dataset` parameter to work properly, the provided file has to have:
- No header
- Comma separated values
- Ref first, Dis second
- Ref and Dis are full paths to the YUV videos

For example:  
path/to/ref.yuv,path/to/dis.yuv

# Result

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

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.


# Acknowledgement

The organization of this repository is inspired by actionformer_release (https://github.com/happyharrycn/actionformer_release/tree/main).