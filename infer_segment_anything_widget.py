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

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_segment_anything.infer_segment_anything_process import InferSegmentAnythingParam

# PyQt GUI framework
from PyQt5.QtWidgets import *

from torch.cuda import is_available

# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferSegmentAnythingWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferSegmentAnythingParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)
       
        self.check_cuda = pyqtutils.append_check(self.grid_layout, "Cuda", self.parameters.cuda and is_available())
        self.check_cuda.setEnabled(is_available())

        self.combo_model_name = pyqtutils.append_combo(self.grid_layout, "Model name")
        self.combo_model_name.addItem("vit_b")
        self.combo_model_name.addItem("vit_l")
        self.combo_model_name.addItem("vit_h")
        self.combo_model_name.setCurrentText(self.parameters.model_name)

        self.spin_points_per_side = pyqtutils.append_spin(self.grid_layout,
                                                          "Points per side",
                                                          self.parameters.points_per_side,
                                                          min=1)

        self.spin_input_size_percent = pyqtutils.append_spin(self.grid_layout,
                                                          "Image size (%)",
                                                          self.parameters.input_size_percent,
                                                          min=1, max=100)       

        
        # self.spin_iou_thres = pyqtutils.append_double_spin(self.grid_layout,
        #                                                    "IoU threshold",
        #                                                   self.parameters.iou_thres,
        #                                                   min=0., max=1.,
        #                                                   step=0.01, decimals=2)
        
        # self.spin_crop_n_layers = pyqtutils.append_spin(self.grid_layout,
        #                                                 "Crop n layer",
        #                                                 self.parameters.crop_n_layers)
        
        # self.spin_crop_n_points_downscale_factor = pyqtutils.append_spin(self.grid_layout,
        #                                                                  "Crop n points downscale factor",
        #                                                                  self.parameters.crop_n_points_downscale_factor)
        # self.spin_min_mask_region_area = pyqtutils.append_spin(self.grid_layout,
        #                                                        "Min max region area",
        #                                                        self.parameters.min_mask_region_area)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_apply(self):
        # Apply button clicked slot
        self.parameters.update = True
        self.parameters.model_name = self.combo_model_name.currentText()
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.points_per_side = self.spin_points_per_side.value()
        self.parameters.input_size_percent = self.spin_input_size_percent.value()
        # self.parameters.spin_iou_thres = self.spin_iou_thres.value()
        # self.parameters.spin_crop_n_layers = self.spin_crop_n_layers.value()
        # self.parameters.crop_n_points_downscale_factor = self.spin_crop_n_points_downscale_factor.value()
        # self.parameters.min_mask_region_area = self.spin_min_mask_region_area.value()



        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferSegmentAnythingWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_segment_anything"

    def create(self, param):
        # Create widget object
        return InferSegmentAnythingWidget(param, None)
