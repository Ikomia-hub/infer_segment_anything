# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import copy
from ikomia import core, dataprocess, utils

import numpy as np
import torch
import cv2
from pathlib import Path
import os
import requests
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from ikomia.dataprocess import CGraphicsInput
from ikomia.core import CGraphicsRectangle


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferSegmentAnythingParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.model_name = "vit_b"
        self.points_per_side = 32
        self.points_per_batch = 64
        self.stability_score_thresh = 0.95
        self.box_nms_thresh = 0.7
        self.iou_thres = 0.88
        self.crop_n_layers = 0
        self.crop_nms_thresh = 0.70
        self.crop_overlap_ratio = float(512 / 1500)
        self.crop_n_points_downscale_factor = 1
        self.min_mask_region_area = 0
        self.input_size_percent = 100
        self.mask_id = 1
        self.cuda = torch.cuda.is_available()
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = param_map["model_name"]
        self.points_per_side = int(param_map["points_per_side"])
        self.points_per_batch = int(param_map["points_per_batch"])
        self.iou_thres = float(param_map["iou_thres"])
        self.stability_score_thresh = float(param_map["stability_score_thresh"])
        self.box_nms_thresh = float(param_map["box_nms_thresh"])
        self.crop_n_layers = int(param_map["crop_n_layers"])
        self.crop_nms_thresh = float(param_map["crop_nms_thresh"])
        self.crop_overlap_ratio = float(param_map["crop_overlap_ratio"])
        self.crop_n_points_downscale_factor = int(param_map["crop_n_points_downscale_factor"])
        self.min_mask_region_area = int(param_map["min_mask_region_area"])
        self.input_size_percent = int(param_map["input_size_percent"])
        self.mask_id = int(param_map["mask_id"])
        self.cuda = utils.strtobool(param_map["cuda"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["model_name"] = self.model_name
        param_map["points_per_side"] = str(self.points_per_side)
        param_map["points_per_batch"] = str(self.points_per_batch)
        param_map["iou_thres"] = str(self.iou_thres)
        param_map["stability_score_thresh"] = str(self.stability_score_thresh)
        param_map["box_nms_thresh"] = str(self.box_nms_thresh)
        param_map["crop_n_layers"] = str(self.crop_n_layers)
        param_map["crop_overlap_ratio"] = str(self.crop_overlap_ratio)
        param_map["crop_nms_thresh"] = str(self.crop_nms_thresh)
        param_map["crop_n_points_downscale_factor"] = str(self.crop_n_points_downscale_factor)
        param_map["min_mask_region_area"] = str(self.min_mask_region_area)
        param_map["input_size_percent"] = str(self.input_size_percent)
        param_map["mask_id"] = str(self.mask_id)
        param_map["cuda"] = str(self.cuda)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferSegmentAnything(dataprocess.CSemanticSegmentationTask):

    def __init__(self, name, param):
        dataprocess.CSemanticSegmentationTask.__init__(self, name)
        #self.add_output(dataprocess.CGraphicsOutput())
        # Create parameters class
        if param is None:
            self.set_param_object(InferSegmentAnythingParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.mask_generator = None
        self.input_point = None
        self.input_label = None
        self.input_box = []
        self.multi_mask_out = True
        self.device = torch.device("cpu")
        self.base_url= "https://dl.fbaipublicfiles.com/segment_anything/"

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1
    
    def get_model(self, model_name):
        model_list = {"vit_b": "sam_vit_b_01ec64.pth",
                      "vit_l": "sam_vit_l_0b3195.pth",
                      "vit_h": "sam_vit_h_4b8939.pth"}

        model_folder = Path(os.path.dirname(os.path.realpath(__file__)) + "/models/")
        model_weight =  os.path.join(str(model_folder), model_list[model_name])

        if not os.path.isfile(model_weight):
            Path(model_folder).mkdir(parents=True, exist_ok=True)
            print("Downloading the model...")
            model_url = self.base_url + model_list[model_name]
            response = requests.get(model_url)
            to_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
            with open(os.path.join(to_folder, model_list[model_name]) , 'wb') as f:
                f.write(response.content)
        return model_weight
    
    def infer_mask_generator(self, image, sam_model):
        # Get parameters :
        param = self.get_param_object()

        mask_generator = SamAutomaticMaskGenerator(
                                model=sam_model,
                                points_per_side= param.points_per_side, # number of points to be sampled along one side of the image
                                points_per_batch=param.points_per_batch, # number of points to be sampled in one batch
                                pred_iou_thresh=param.iou_thres, # predicted mask quality
                                stability_score_thresh= param.stability_score_thresh, #  cutoff used to binarize the mask predictions
                                box_nms_thresh=param.box_nms_thresh, # box IoU cutoff (filter duplicate masks)
                                crop_n_layers = param.crop_n_layers, #  mask prediction will be run again on crops of the image
                                crop_overlap_ratio=param.crop_overlap_ratio,
                                crop_nms_thresh=param.crop_nms_thresh,
                                crop_n_points_downscale_factor= param.crop_n_points_downscale_factor,
                                min_mask_region_area=param.min_mask_region_area # post-process remove disconected regions
                                    )

        # Generate mask
        results = mask_generator.generate(image)

        if len(results) > 0:
            mask_output = np.zeros((
                        results[0]["segmentation"].shape[0],
                        results[0]["segmentation"].shape[1]
                        ))
            for i, mask_bool in enumerate(results):
                i += 1
                mask_output = mask_output + mask_bool["segmentation"] * i

        else:
            print("No mask predicted, increasing the number of points per side may help")

        return mask_output

    def infer_predictor(self, graph_input,src_image, resizing, sam_model):
        # Get parameters :
        param = self.get_param_object()

        graphics = graph_input.get_items() #Get list of input graphics items.

        self.box = []
        for i, graphic in enumerate(graphics):
            bboxes = graphics[i].get_bounding_rect() # Get graphic coordinates
            if graphic.get_type() == 3: # rectangle
                x1 = bboxes[0]*resizing
                y1 = bboxes[1]*resizing
                x2 = (bboxes[2]+bboxes[0])*resizing
                y2 = (bboxes[3]+bboxes[1])*resizing
                self.box.append([x1, y1, x2, y2])
                self.input_box = np.array(self.box)
                self.multi_mask_out = False
                
            if graphic.get_type() == 1: # point
                point = [bboxes[0]*resizing, bboxes[1]*resizing]
                self.input_point = np.array([point])
                self.input_label = np.array([1])
                self.multi_mask_out = True
                self.input_box = None

        predictor = SamPredictor(sam_model)
        # Calculate the necesssary image embedding
        predictor.set_image(src_image)

        # Inference 
        if self.input_box is not None and len(self.input_box) > 0: # Inference from multiple boxes
            self.multi_mask_out = False
            input_boxes = torch.tensor(self.input_box, device=self.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(
                                                input_boxes, 
                                                src_image.shape[:2]
                                                )
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=self.multi_mask_out,
                )

            mask_output = np.zeros((
                            src_image.shape[0],
                            src_image.shape[1]
                            ))

            for i, mask_bool in enumerate(masks):
                mask = mask_bool.cpu().numpy()[0]
                i += 1
                mask_output = mask_output + mask * i
        
        else: # Inference from a single point or box
            masks, _, _ = predictor.predict(
                point_coords=self.input_point,
                point_labels=self.input_label,
                box=self.input_box,
                multimask_output=self.multi_mask_out,
            )

        # Select the mask to be displayed
        if self.multi_mask_out is True:
            mask_output = masks[param.mask_id-1]
        else:
            mask_output = mask_output

        return mask_output

    def run(self):
        # Core function of your process
        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Get input
        task_input = self.get_input(0)

        # Get parameters :
        param = self.get_param_object()

        # Get image from input/output (numpy array):
        src_image = task_input.get_image()

        # Resize image
        ratio = param.input_size_percent / 100
        h_orig, w_orig = src_image.shape[0], src_image.shape[1]
        if param.input_size_percent < 100:
            width = int(src_image.shape[1] * ratio)
            height = int(src_image.shape[0] * ratio)
            dim = (width, height)
            src_image = cv2.resize(src_image, dim, interpolation = cv2.INTER_LINEAR)

        # Load model
        if param.update or self.mask_generator is None:
            self.device = torch.device("cuda") if param.cuda else torch.device("cpu")
            model_path = self.get_model(param.model_name)
            sam = sam_model_registry[param.model_name](checkpoint=model_path)
            sam.to(device=self.device)

        graph_input = self.get_input(1)
        if graph_input.is_data_available():
            mask = self.infer_predictor(graph_input, src_image, ratio, sam)
        else:
            mask = self.infer_mask_generator(src_image, sam)

        mask = mask.astype("uint8")
        if param.input_size_percent < 100:
            mask = cv2.resize(
                            mask,
                            (w_orig, h_orig),
                            interpolation = cv2.INTER_NEAREST
                                )

        self.get_output(0)
        self.set_mask(mask)

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferSegmentAnythingFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_segment_anything"
        self.info.short_description = "Inference for Segment Anything Model (SAM)."
        self.info.description = "This algorithm proposes inference for the Segment Anything Model (SAM). " \
                                "It can be used to generate masks for all objects in an image."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.version = "1.0.0"
        self.info.icon_path = "icons/meta_icon.jpg"
        self.info.authors = "Alexander Kirillov, Alex Berg, Chloe Rolland, Eric Mintun, Hanzi Mao, " \
                            "Laura Gustafson, Nikhila Ravi, Piotr Dollar, Ross Girshick, "  \
                            "Spencer Whitehead, Wan-Yen Lo"
        self.info.article = "Segment Anything"
        self.info.journal = "ArXiv"
        self.info.year = 2023
        self.info.license = "Apache 2.0 license"
        # URL of documentation
        self.info.documentation_link = "https://segment-anything.com/"
        # Code source repository
        self.info.repository = "https://github.com/facebookresearch/segment-anything"
        # Keywords used for search
        self.info.keywords = "your,keywords,here"

    def create(self, param=None):
        # Create process object
        return InferSegmentAnything(self.info.name, param)
