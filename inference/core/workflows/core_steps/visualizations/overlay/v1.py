from typing import List, Literal, Optional, Type, Union
import numpy as np

import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
    VisualizationBlock,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition, WorkflowImageData
from inference.core.workflows.execution_engine.entities.types import (
    BATCH_OF_IMAGES_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    INTEGER_KIND,
    RGB_COLOR_KIND,
    WILDCARD_KIND,
    FloatZeroToOne,
    StepOutputImageSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlock, WorkflowBlockManifest

TYPE: str = "roboflow_core/overlay_visualization@v1"
SHORT_DESCRIPTION = "Overlays an image on a frame."
LONG_DESCRIPTION = """
The `OverlayImage` block overlays an image on a frame
using Supervision's `sv.overlay_image`.
"""


class OverlayManifest(WorkflowBlockManifest):
    type: Literal[f"{TYPE}", "OverlayVisualization"]
    name: str
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Overlay Image",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
        }
    )

    image: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        title="Input Image",
        description="The input image for this step.",
        examples=["$inputs.image"],
        #validation_alias=AliasChoices("image", "images"),
    )
    
    overlay: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        title="Overlay Image",
        description="The overlay image for this step.",
        examples=["$inputs.overlay"],
        #validation_alias=AliasChoices("image", "images"),
    )
    
    x: Union[
        FloatZeroToOne, WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        description="Center X of static crop (relative coordinate 0.0-1.0)",
        examples=[0.3, "$inputs.center_x"],
    )
    y: Union[
        FloatZeroToOne, WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        description="Center Y of static crop (relative coordinate 0.0-1.0)",
        examples=[0.3, "$inputs.center_y"],
    )
    width: Union[
        FloatZeroToOne, WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        description="Width of static crop (relative value 0.0-1.0)",
        examples=[0.3, "$inputs.width"],
    )
    height: Union[
        FloatZeroToOne, WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        description="Height of static crop (relative value 0.0-1.0)",
        examples=[0.3, "$inputs.height"],
    )
    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_IMAGE_KEY,
                kind=[
                    BATCH_OF_IMAGES_KIND,
                ],
            ),
        ]

        
    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class OverlayImageBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}

    
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return OverlayManifest

    def run(
        self,
        image: WorkflowImageData,
        overlay: WorkflowImageData,
        # image_1: WorkflowImageData,
        # image_2: WorkflowImageData,
    ) -> BlockResult:

        keep_aspect_ratio = True
        target_resolution_wh = (498, 136)
        x, y = 0, 0
        
        scaled_overlay = sv.resize_image(image=overlay.numpy_image.copy(), resolution_wh=target_resolution_wh, keep_aspect_ratio=keep_aspect_ratio)
        print(overlay.numpy_image.shape)
        scaled_overlay_shape = scaled_overlay.shape

        rect = sv.Rect(x, y, width=target_resolution_wh[0], height=target_resolution_wh[1])
        
        annotated_image = sv.draw_image(
            scene=image.numpy_image. copy(), 
            image=overlay.numpy_image.copy(),
            rect=rect,
            opacity=1 
        )

        output = WorkflowImageData(
            parent_metadata=image.parent_metadata,
            workflow_root_ancestor_metadata=image.workflow_root_ancestor_metadata,
            numpy_image=annotated_image,
        )

        return {OUTPUT_IMAGE_KEY: output}
        # pass
