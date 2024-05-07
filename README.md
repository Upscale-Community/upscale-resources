This is a simple repository meant to document current training software and tools to go along with it

## Resources

### Training Software:
#### Recommended software:
- [neosr](https://github.com/muslll/neosr): NeoSR is a training software that prioritizes stability and performance. This release utlizies Pytorch 2.0, adds AMP & torch.compile() support, and much more. This is the easiest to start with and is consistently updated
- [neosr-extended](https://github.com/Upscale-Community/neosr-extended): NeoSR-extended is a custom fork of NeoSR that adds additional functions/features. It is updated consistently with the original fork. Extended retains as many features as possible that have been removed from the original neosr, mostly architectures.
- [sudo's traiNNer](https://github.com/styler00dollar/Colab-traiNNer/): This is custom training software supports a significant amount of losses and architectures, and is very versatile. However, it is much more complex to use

#### Other training software (not recommended)

- [traiNNer-redux](https://github.com/joeyballentine/traiNNer-redux): traiNNer-redux is a recently forked version of BasicSR with additional losses such as color and contextual loss  

- [traiNNer-redux-FJ](https://github.com/FlotingDream/traiNNer-redux): A fork of traiNNer-redux by @FloatingJoy#0260 that has additional arch support

- [BasicSR](https://github.com/XPixelGroup/BasicSR): The official training software for many architectures such as ESRGAN and SwinIR

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN): Very similar to BasicSR with a focus on the Real-ESRGAN architecture, including compact models

- [KAIR](https://github.com/cszn/KAIR): Multifunctional training software that supports many arches

### Communities:
- [Enhance Everything!](https://discord.gg/cpAUpDK): A community focused on training models for various purposes, including game upscaling, anime, photos, and more! Look in the `#training` channel's pinned messages for a bunch more info

### Tools:
- [OpenModelDB](https://openmodeldb.info): This website contains a bunch of community trained models. You can use these models as they are, or use them as pretrains for your own model
- [chaiNNer](https://chainner.app/): This is a tool that can be used to degrade datasets as well, however it also supports *many* other functions, including using the models you've trained
- [AnimeJaNaiConverterGui](https://github.com/the-database/AnimeJaNaiConverterGui): Allows for fast video upscaling utilizing TensorRT (On Nvidia cards), DirectML, or NCNN within a clean GUI. Only supports ONNX. Use chaiNNer to convert pth models to ONNX for usage.
- [Kim's Helpful Scripts](https://github.com/Kim2091/helpful-scripts): This is a collection of scripts I've made to assist with using training software. This includes:
   * A script to efficiently tile your datasets to speed up training
   * a "Dataset destroyer" to manually degrade images
   * and more!
- [ImgAlign](https://github.com/sonic41592/ImgAlign): This is a great tool to automatically align your datasets. It supports TPS warping, which is a great feature for difficult image pairs
- [img-ab](https://github.com/the-database/img-ab): An image comparison tool that is lightweight and efficient with a lot of options. This can be helpful when determining progress on your model
- [Simple Image Compare](https://github.com/Sirosky/Simple-Image-Compare): A basic tool for comparing images
- [Image Pearer](https://github.com/Sirosky/Image-Pearer): This tool will create image pairs from a given source. It looks at two folders of images (one HR, one LR), and automatically matches them up

### Tutorials:
Constantly updated: [Sirosky's NeoSR Training Guide](https://github.com/Sirosky/Upscale-Hub/wiki/%F0%9F%93%88-Training-a-Model-in-NeoSR)

Extra Video Tutorials:

*1 and 2 are outdated, but much of the info applies to [NeoSR](https://github.com/muslll/neosr)*
- Preparing for, training, and releasing a model with neosr: https://www.youtube.com/watch?v=8XUHbeE8prU
- Training a model with Real-ESRGAN Compact: https://www.youtube.com/watch?v=l_tZE5l90VY
- Training an image upscaling model: https://www.youtube.com/watch?v=iH7-eYlf7eg
- Dataset preparation: https://www.youtube.com/watch?v=TBiVIzQkptI

### Guidelines for training:

<details>
<summary>Descriptions of each loss available in most training software</summary>
Here is a brief summary of some of the loss functions that are used for super resolution and image restoration tasks. Thanks to korvin for the info!

1. **L1Loss:** This is the mean absolute error (MAE) between the predicted and target images. It measures the average pixel-wise difference, and is simple to implement and fast to compute. However, it may produce blurry results and does not account for perceptual quality or high-frequency details. It can be used for any type of image. For example, it is suitable for low-level tasks such as denoising or inpainting, but also super resolution. It can be combined with other losses such as perceptual loss or GAN loss to improve the results.

2. **LRGBLoss:** This is a variant of L1Loss that computes the MAE separately for each color channel (red, green, blue) and then averages them. It is similar to L1Loss in terms of advantages and disadvantages, but it may be more sensitive to color differences. It can be used for any type of images, but it may not be optimal for grayscale images or images with different color spaces3.

3. **PerceptualLoss:** This is a loss function that uses a pre-trained network, such as VGG, to extract high-level features from the predicted and target images and then computes the MAE (or other measures) between them. It aims to capture the perceptual similarity and semantic content of the images, rather than the pixel-wise difference. It can produce more natural and realistic results, especially for high-level tasks such as super resolution or style transfer. However, it is computationally expensive, requires regularization and hyper-parameter tuning, and involves a large network trained on an unrelated task. It can be used for any type of images, but it may not be optimal for low-level tasks or images with different domains24.

4. **ContextualLoss:** This is a loss function that measures the similarity between two images based on the distribution of local patches. It uses a cosine similarity metric to compare the patches and then aggregates them using a generalized mean function. It can capture both global and local structures, as well as texture and style information. It can produce more diverse and detailed results, especially for texture synthesis or style transfer. However, it is computationally expensive, requires patch size selection and normalization, and may not be robust to geometric transformations or occlusions. It can be used for any type of images, but it may not be optimal for images with large variations or complex semantics5.

5. **ColorLoss:** There are many types of color loss. An explicit example would be this: a loss function that measures the color difference between two images using the CIEDE2000 formula, which is based on the human perception of color and accounts for factors such as luminance, hue, chroma, and contrast. It can produce more accurate and consistent color reproduction, especially for color enhancement or correction. However, it is computationally expensive, requires color space conversion and calibration, and may not capture other aspects of image quality such as sharpness or noise. It can be used for any type of images, but it may not be optimal for grayscale images or images with different color spaces.

6. **AverageLoss:** This is a loss function that computes the average of multiple loss functions, such as L1Loss, PerceptualLoss, ColorLoss, etc. It can combine the advantages of different losses and balance their trade-offs. It can produce more comprehensive and satisfactory results, especially for multi-objective tasks such as super resolution with color enhancement. It is very lightweight, but with some implementations can require fine-tuning. It can be used for any type of images, but it may not be optimal for single-objective tasks or tasks with conflicting objectives.

7. **GANLoss:** This is a loss function that uses a generative adversarial network (GAN) to discriminate between the predicted and target images. It aims to fool the discriminator network into thinking that the predicted image is real and indistinguishable from the target image. It can produce more sharp and realistic results, especially for high-level tasks such as super resolution or style transfer. However, it is computationally expensive, requires careful design and training of the discriminator network, and may suffer from instability or mode collapse issues. It can be used for any type of images, but it may not be optimal for low-level tasks or tasks with limited data.
</details>

<details>
<summary>Loss information, where to aim to have your loss values:</summary>

- **Most Losses:**: Aim for a value of 0. Lower is better.
- **GAN**: Ideal value varies with implementation.
- **SSIM**: Aim for a value of 1. Higher is better.

Metrics:
- **PSNR**: No specific target value. Higher is better.

**Example:**
- A loss value of 4.1821e-04 (0.00041821 in decimal) is better than 4.1821e-01 (0.41821) for the main losses. A value closer to 0 is ideal in this scenario.
- A loss value of 2.5325e+03 (2532.5 in decimal) is considered bad, as it's very high. You should tweak your config accordingly.

</details>

