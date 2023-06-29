# infer_segment_anything

The Segment Anything Model (SAM) offers multiple inference modes for generating masks:
1. Automated mask generation (segmentation over the full image)
2. Segmentation masks from prompts (bounding boxes or point)

The SAM model can be loaded with three different encoders: ViT-B, ViT-L, and ViT-H. The encoders differ in parameter counts, with ViT-B containing 91M, ViT-L containing 308M, and ViT-H containing 636M parameters. Keep in mind that encoder size also affects inference speed, so choose the appropriate encoder based on your specific use case.

## 1. Automated mask generation
When no prompt is used, SAM will generate masks automatically over the entire image. 
You can select the number of masks using the parameter "Points per side" on Ikomia STUDIO or "points_per_side" with the API. Here is an example with ViT-H using the default settings (32 points/side).  

<img src="images/dog_auto_seg.png"  width="30%" height="30%">


## 2. Segmentation mask with graphic prompts:
Given a graphic prompts: a single point or boxes SAM can predict masks over the desired objects. 
- Ikomia API: Add the parameter "image_path = PATH/TO/YOUR/IMAGE"  to draw over the image.
    - Point: A point can be generated with a left click
    - Box: Left click > drag > release

- Ikomia STUDIO: Open the Toggle graphics toolbar 
    - Point: Select the point tool
    - Box: Select the Square/Rectangle tool

### a. Single point 
SAM with generate three outputs given a single point (3 best scores). 
You can select which mask to output using the mask_id parameters (1, 2 or 3) 
<img src="images/dog_single_point.png"  width="80%" height="80%">

### b. Boxes
A single point can be ambiguous, drawing a box over the desired object usually output a mask closer to expectation. 

SAM can also take multiple inputs prompts.

<img src="images/cats_boxes.png"  width="80%" height="80%">

### c. Point and box

Point and box can be combined by including both types of prompts to the predictor. Here this can be used to select just the trucks's tire, instead of the entire wheel.

<img src="images/truck_box_point.png"  width="80%" height="80%">