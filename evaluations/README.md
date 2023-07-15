# Evaluations
To compare different generative models, we use FID, Precision, Recall, and Inception Score.
These metrics can all be calculated using batches of samples, which we store in `.npz` (numpy) files.
We use code from [openai/guided-diffusion](https://github.com/openai/guided-diffusion) for evaluation.

# Download batches
We use pre-computed reference images of [openai/guided-diffusion](https://github.com/openai/guided-diffusion), which are stored as `.npz` format.
Reference dataset batches contain pre-computed statistics over the whole dataset, as well as 10,000 images for computing Precision and Recall. All other batches contain 50,000 images which can be used to compute statistics and Precision/Recall.

Here is reference samples for evaluation as same in [openai/guided-diffusion](https://github.com/openai/guided-diffusion).
* ImageNet 256x256: [reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz)

# Run evaluations

First, generate or download a batch of samples and download the corresponding reference batch for the given dataset. For this example, we'll use ImageNet 256x256, so the refernce batch is `VIRTUAL_imagenet256_labeled.npz` and we can use the sample batch `admnet_guided_upsampled_imagenet256.npz`.

Next, run the `evaluator.py` script. The requirements of this script can be found in [requirements.txt](requirements.txt). Pass two arguments to the script: the reference batch and the sample batch. The script will download the InceptionV3 model used for evaluations into the current working directory (if it is not already present). This file is roughly 100MB.

The output of the script will look something like this, where the first `...` is a bunch of verbose TensorFlow logging:

```
$ python evaluator.py --ref_batch VIRTUAL_imagenet256_labeled.npz --sample_batch [path_for_generated_images.npz] --save_result_path [File path for logging results]
...
computing reference batch activations...
computing/reading reference batch statistics...
computing sample batch activations...
computing/reading sample batch statistics...
Computing evaluations...
Inception Score: [Inception score value]
FID: [FID score value]
Precision: [Precision score value]
Recall: [Recall score value]
```
