import statistics
from collections import Counter, defaultdict
from enum import Enum
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from typing_extensions import Annotated

from inference.enterprise.workflows.complier.steps_executors.constants import (
    DETECTION_ID_KEY,
    HEIGHT_KEY,
    PARENT_ID_KEY,
    WIDTH_KEY,
)
from inference.enterprise.workflows.core_steps.common.utils import detection_to_xyxy
from inference.enterprise.workflows.entities.steps import OutputDefinition
from inference.enterprise.workflows.entities.types import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_METADATA_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    PARENT_ID_KIND,
    PREDICTION_TYPE_KIND,
    FloatZeroToOne,
    FlowControl,
    InferenceParameterSelector,
    StepOutputSelector,
)
from inference.enterprise.workflows.errors import ExecutionGraphError
from inference.enterprise.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)


class AggregationMode(Enum):
    AVERAGE = "average"
    MAX = "max"
    MIN = "min"


LONG_DESCRIPTION = """
Combine detections from multiple detection-based models based on a majority vote 
strategy.

This block is useful if you have multiple specialized models that you want to consult 
to determine whether a certain object is present in an image.

See the table below to explore the values you can use to configure the consensus block.
"""

SHORT_DESCRIPTION = (
    "Combine predictions from multiple detections models to make a "
    "decision about object presence."
)


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "fusion",
        }
    )
    type: Literal["DetectionsConsensus"]
    predictions: List[
        StepOutputSelector(
            kind=[
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
                KEYPOINT_DETECTION_PREDICTION_KIND,
            ]
        ),
    ] = Field(
        min_items=1,
        description="Reference to detection-like model predictions made against single image to agree on model consensus",
        examples=[["$steps.a.predictions", "$steps.b.predictions"]],
    )
    image_metadata: Annotated[
        StepOutputSelector(kind=[IMAGE_METADATA_KIND]),
        Field(
            description="Metadata of image used to create `predictions`. Must be output from the step referred in `predictions` field",
            examples=["$steps.detection.image"],
        ),
    ]
    required_votes: Union[
        PositiveInt, InferenceParameterSelector(kind=[INTEGER_KIND])
    ] = Field(
        description="Required number of votes for single detection from different models to accept detection as output detection",
        examples=[2, "$inputs.required_votes"],
    )
    class_aware: Union[bool, InferenceParameterSelector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Flag to decide if margin detections is class-aware or only bounding boxes aware",
        examples=[True, "$inputs.class_aware"],
    )
    iou_threshold: Union[
        FloatZeroToOne, InferenceParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        default=0.3,
        description="IoU threshold to consider detections from different models as matching (increasing votes for region)",
        examples=[0.3, "$inputs.iou_threshold"],
    )
    confidence: Union[
        FloatZeroToOne, InferenceParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        default=0.0,
        description="Confidence threshold for merged detections",
        examples=[0.1, "$inputs.confidence"],
    )
    classes_to_consider: Optional[
        Union[List[str], InferenceParameterSelector(kind=[LIST_OF_VALUES_KIND])]
    ] = Field(
        default=None,
        description="Optional list of classes to consider in consensus procedure.",
        examples=[["a", "b"], "$inputs.classes_to_consider"],
    )
    required_objects: Optional[
        Union[
            PositiveInt,
            Dict[str, PositiveInt],
            InferenceParameterSelector(kind=[INTEGER_KIND, DICTIONARY_KIND]),
        ]
    ] = Field(
        default=None,
        description="If given, it holds the number of objects that must be present in merged results, to assume that object presence is reached. Can be selector to `InferenceParameter`, integer value or dictionary with mapping of class name into minimal number of merged detections of given class to assume consensus.",
        examples=[3, {"a": 7, "b": 2}, "$inputs.required_objects"],
    )
    presence_confidence_aggregation: AggregationMode = Field(
        default=AggregationMode.MAX,
        description="Mode dictating aggregation of confidence scores and classes both in case of object presence deduction procedure.",
        examples=["max", "min"],
    )
    detections_merge_confidence_aggregation: AggregationMode = Field(
        default=AggregationMode.AVERAGE,
        description="Mode dictating aggregation of confidence scores and classes both in case of boxes consensus procedure. One of `average`, `max`, `min`. Default: `average`. While using for merging overlapping boxes, against classes - `average` equals to majority vote, `max` - for the class of detection with max confidence, `min` - for the class of detection with min confidence.",
        examples=["min", "max"],
    )
    detections_merge_coordinates_aggregation: AggregationMode = Field(
        default=AggregationMode.AVERAGE,
        description="Mode dictating aggregation of bounding boxes. One of `average`, `max`, `min`. Default: `average`. `average` means taking mean from all boxes coordinates, `min` - taking smallest box, `max` - taking largest box.",
        examples=["min", "max"],
    )


class DetectionsConsensusBlock(WorkflowBlock):

    @classmethod
    def get_input_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(
                name="predictions",
                kind=[OBJECT_DETECTION_PREDICTION_KIND],
            ),
            OutputDefinition(name="image", kind=[IMAGE_METADATA_KIND]),
            OutputDefinition(
                name="object_present", kind=[BOOLEAN_KIND, DICTIONARY_KIND]
            ),
            OutputDefinition(
                name="presence_confidence",
                kind=[FLOAT_ZERO_TO_ONE_KIND, DICTIONARY_KIND],
            ),
            OutputDefinition(name="predictions_type", kind=[PREDICTION_TYPE_KIND]),
        ]

    async def run_locally(
        self,
        predictions: List[List[List[dict]]],
        image_metadata: List[dict],
        required_votes: int,
        class_aware: bool,
        iou_threshold: float,
        confidence: float,
        classes_to_consider: Optional[List[str]],
        required_objects: Optional[Union[int, Dict[str, int]]],
        presence_confidence_aggregation: AggregationMode,
        detections_merge_confidence_aggregation: AggregationMode,
        detections_merge_coordinates_aggregation: AggregationMode,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        if len(predictions) < 1:
            raise ExecutionGraphError(
                f"Consensus step requires at least one source of predictions."
            )
        batch_sizes = get_and_validate_batch_sizes(
            all_predictions=predictions,
        )
        batch_size = batch_sizes[0]
        results = []
        for batch_index in range(batch_size):
            batch_element_predictions = [e[batch_index] for e in predictions]
            (
                parent_id,
                object_present,
                presence_confidence,
                consensus_detections,
            ) = resolve_batch_consensus(
                predictions=batch_element_predictions,
                required_votes=required_votes,
                class_aware=class_aware,
                iou_threshold=iou_threshold,
                confidence=confidence,
                classes_to_consider=classes_to_consider,
                required_objects=required_objects,
                presence_confidence_aggregation=presence_confidence_aggregation,
                detections_merge_confidence_aggregation=detections_merge_confidence_aggregation,
                detections_merge_coordinates_aggregation=detections_merge_coordinates_aggregation,
            )
            results.append(
                {
                    "predictions": consensus_detections,
                    "parent_id": parent_id,
                    "object_present": object_present,
                    "presence_confidence": presence_confidence,
                    "image": image_metadata[batch_index],
                    "prediction_type": "object-detection",
                }
            )
        return results


def get_and_validate_batch_sizes(
    all_predictions: List[List[List[dict]]],
) -> List[int]:
    batch_sizes = get_predictions_batch_sizes(all_predictions=all_predictions)
    if not all_batch_sizes_equal(batch_sizes=batch_sizes):
        raise ExecutionGraphError(f"Detected missmatch of input dimensions.")
    return batch_sizes


def get_predictions_batch_sizes(all_predictions: List[List[List[dict]]]) -> List[int]:
    return [len(predictions) for predictions in all_predictions]


def does_not_detected_objects_in_any_source(predictions: List[List[dict]]) -> bool:
    return all(len(p) == 0 for p in predictions)


def get_parent_id_of_predictions_from_different_sources(
    predictions: List[List[dict]],
) -> str:
    encountered_parent_ids = {
        p[PARENT_ID_KEY] for prediction_source in predictions for p in prediction_source
    }
    if len(encountered_parent_ids) > 1:
        raise ExecutionGraphError(
            f"Missmatch in predictions - while executing consensus step, "
            f"in equivalent batches, detections are assigned different parent "
            f"identifiers, whereas consensus can only be applied for predictions "
            f"made against the same input."
        )
    return list(encountered_parent_ids)[0]


def filter_predictions(
    predictions: List[List[dict]],
    classes_to_consider: Optional[List[str]],
) -> List[List[dict]]:
    if classes_to_consider is None:
        return predictions
    classes_to_consider = set(classes_to_consider)
    return [
        [
            detection
            for detection in detections
            if detection["class"] in classes_to_consider
        ]
        for detections in predictions
    ]


def get_detections_from_different_sources_with_max_overlap(
    detection: dict,
    source: int,
    predictions: List[List[dict]],
    iou_threshold: float,
    class_aware: bool,
    detections_already_considered: Set[str],
) -> Dict[int, Tuple[dict, float]]:
    current_max_overlap = {}
    for other_source, other_detection in enumerate_detections(
        predictions=predictions,
        excluded_source=source,
    ):
        if other_detection[DETECTION_ID_KEY] in detections_already_considered:
            continue
        if class_aware and detection["class"] != other_detection["class"]:
            continue
        iou_value = calculate_iou(
            detection_a=detection,
            detection_b=other_detection,
        )
        if iou_value <= iou_threshold:
            continue
        if current_max_overlap.get(other_source) is None:
            current_max_overlap[other_source] = (other_detection, iou_value)
        if current_max_overlap[other_source][1] < iou_value:
            current_max_overlap[other_source] = (other_detection, iou_value)
    return current_max_overlap


def enumerate_detections(
    predictions: List[List[dict]],
    excluded_source: Optional[int] = None,
) -> Generator[Tuple[int, dict], None, None]:
    for source_id, detections in enumerate(predictions):
        if excluded_source is not None and excluded_source == source_id:
            continue
        for detection in detections:
            yield source_id, detection


def calculate_iou(detection_a: dict, detection_b: dict) -> float:
    box_a = detection_to_xyxy(detection=detection_a)
    box_b = detection_to_xyxy(detection=detection_b)
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    intersection = max(0, x_b - x_a) * max(0, y_b - y_a)
    bbox_a_area, bbox_b_area = get_detection_sizes(
        detections=[detection_a, detection_b]
    )
    union = float(bbox_a_area + bbox_b_area - intersection)
    if union == 0.0:
        return 0.0
    return intersection / float(bbox_a_area + bbox_b_area - intersection)


def all_batch_sizes_equal(batch_sizes: List[int]) -> bool:
    if len(batch_sizes) == 0:
        return True
    reference = batch_sizes[0]
    return all(e == reference for e in batch_sizes)


def resolve_batch_consensus(
    predictions: List[List[dict]],
    required_votes: int,
    class_aware: bool,
    iou_threshold: float,
    confidence: float,
    classes_to_consider: Optional[List[str]],
    required_objects: Optional[Union[int, Dict[str, int]]],
    presence_confidence_aggregation: AggregationMode,
    detections_merge_confidence_aggregation: AggregationMode,
    detections_merge_coordinates_aggregation: AggregationMode,
) -> Tuple[str, bool, Dict[str, float], List[dict]]:
    if does_not_detected_objects_in_any_source(predictions=predictions):
        return "undefined", False, {}, []
    parent_id = get_parent_id_of_predictions_from_different_sources(
        predictions=predictions,
    )
    predictions = filter_predictions(
        predictions=predictions,
        classes_to_consider=classes_to_consider,
    )
    detections_already_considered = set()
    consensus_detections = []
    for source_id, detection in enumerate_detections(predictions=predictions):
        (
            consensus_detections_update,
            detections_already_considered,
        ) = get_consensus_for_single_detection(
            detection=detection,
            source_id=source_id,
            predictions=predictions,
            iou_threshold=iou_threshold,
            class_aware=class_aware,
            required_votes=required_votes,
            confidence=confidence,
            detections_merge_confidence_aggregation=detections_merge_confidence_aggregation,
            detections_merge_coordinates_aggregation=detections_merge_coordinates_aggregation,
            detections_already_considered=detections_already_considered,
        )
        consensus_detections += consensus_detections_update
    (
        object_present,
        presence_confidence,
    ) = check_objects_presence_in_consensus_predictions(
        consensus_detections=consensus_detections,
        aggregation_mode=presence_confidence_aggregation,
        class_aware=class_aware,
        required_objects=required_objects,
    )
    return (
        parent_id,
        object_present,
        presence_confidence,
        consensus_detections,
    )


def get_consensus_for_single_detection(
    detection: dict,
    source_id: int,
    predictions: List[List[dict]],
    iou_threshold: float,
    class_aware: bool,
    required_votes: int,
    confidence: float,
    detections_merge_confidence_aggregation: AggregationMode,
    detections_merge_coordinates_aggregation: AggregationMode,
    detections_already_considered: Set[str],
) -> Tuple[List[dict], Set[str]]:
    if detection["detection_id"] in detections_already_considered:
        return ([], detections_already_considered)
    consensus_detections = []
    detections_with_max_overlap = (
        get_detections_from_different_sources_with_max_overlap(
            detection=detection,
            source=source_id,
            predictions=predictions,
            iou_threshold=iou_threshold,
            class_aware=class_aware,
            detections_already_considered=detections_already_considered,
        )
    )
    if len(detections_with_max_overlap) < (required_votes - 1):
        return consensus_detections, detections_already_considered
    detections_to_merge = [detection] + [
        matched_value[0] for matched_value in detections_with_max_overlap.values()
    ]
    merged_detection = merge_detections(
        detections=detections_to_merge,
        confidence_aggregation_mode=detections_merge_confidence_aggregation,
        boxes_aggregation_mode=detections_merge_coordinates_aggregation,
    )
    if merged_detection["confidence"] < confidence:
        return consensus_detections, detections_already_considered
    consensus_detections.append(merged_detection)
    detections_already_considered.add(detection[DETECTION_ID_KEY])
    for matched_value in detections_with_max_overlap.values():
        detections_already_considered.add(matched_value[0][DETECTION_ID_KEY])
    return consensus_detections, detections_already_considered


def check_objects_presence_in_consensus_predictions(
    consensus_detections: List[dict],
    class_aware: bool,
    aggregation_mode: AggregationMode,
    required_objects: Optional[Union[int, Dict[str, int]]],
) -> Tuple[bool, Dict[str, float]]:
    if len(consensus_detections) == 0:
        return False, {}
    if required_objects is None:
        required_objects = 0
    if issubclass(type(required_objects), dict) and not class_aware:
        required_objects = sum(required_objects.values())
    if (
        issubclass(type(required_objects), int)
        and len(consensus_detections) < required_objects
    ):
        return False, {}
    if not class_aware:
        aggregated_confidence = aggregate_field_values(
            detections=consensus_detections,
            field="confidence",
            aggregation_mode=aggregation_mode,
        )
        return True, {"any_object": aggregated_confidence}
    class2detections = defaultdict(list)
    for detection in consensus_detections:
        class2detections[detection["class"]].append(detection)
    if issubclass(type(required_objects), dict):
        for requested_class, required_objects_count in required_objects.items():
            if len(class2detections[requested_class]) < required_objects_count:
                return False, {}
    class2confidence = {
        class_name: aggregate_field_values(
            detections=class_detections,
            field="confidence",
            aggregation_mode=aggregation_mode,
        )
        for class_name, class_detections in class2detections.items()
    }
    return True, class2confidence


def merge_detections(
    detections: List[dict],
    confidence_aggregation_mode: AggregationMode,
    boxes_aggregation_mode: AggregationMode,
) -> dict:
    class_name, class_id = AGGREGATION_MODE2CLASS_SELECTOR[confidence_aggregation_mode](
        detections
    )
    x, y, width, height = AGGREGATION_MODE2BOXES_AGGREGATOR[boxes_aggregation_mode](
        detections
    )
    return {
        PARENT_ID_KEY: detections[0][PARENT_ID_KEY],
        DETECTION_ID_KEY: f"{uuid4()}",
        "class": class_name,
        "class_id": class_id,
        "confidence": aggregate_field_values(
            detections=detections,
            field="confidence",
            aggregation_mode=confidence_aggregation_mode,
        ),
        "x": x,
        "y": y,
        "width": width,
        "height": height,
    }


def get_majority_class(detections: List[dict]) -> Tuple[str, int]:
    class_counts = Counter(d["class"] for d in detections)
    most_common_class_name = class_counts.most_common(1)[0][0]
    class_id = [
        d["class_id"] for d in detections if d["class"] == most_common_class_name
    ][0]
    return most_common_class_name, class_id


def get_class_of_most_confident_detection(detections: List[dict]) -> Tuple[str, int]:
    max_confidence = aggregate_field_values(
        detections=detections,
        field="confidence",
        aggregation_mode=AggregationMode.MAX,
    )
    most_confident_prediction = [
        d for d in detections if d["confidence"] == max_confidence
    ][0]
    return most_confident_prediction["class"], most_confident_prediction["class_id"]


def get_class_of_least_confident_detection(detections: List[dict]) -> Tuple[str, int]:
    max_confidence = aggregate_field_values(
        detections=detections,
        field="confidence",
        aggregation_mode=AggregationMode.MIN,
    )
    most_confident_prediction = [
        d for d in detections if d["confidence"] == max_confidence
    ][0]
    return most_confident_prediction["class"], most_confident_prediction["class_id"]


AGGREGATION_MODE2CLASS_SELECTOR = {
    AggregationMode.MAX: get_class_of_most_confident_detection,
    AggregationMode.MIN: get_class_of_least_confident_detection,
    AggregationMode.AVERAGE: get_majority_class,
}


def get_average_bounding_box(detections: List[dict]) -> Tuple[int, int, int, int]:
    x = round(aggregate_field_values(detections=detections, field="x"))
    y = round(aggregate_field_values(detections=detections, field="y"))
    width = round(aggregate_field_values(detections=detections, field="width"))
    height = round(aggregate_field_values(detections=detections, field="height"))
    return x, y, width, height


def get_smallest_bounding_box(detections: List[dict]) -> Tuple[int, int, int, int]:
    detection_sizes = get_detection_sizes(detections=detections)
    smallest_size = min(detection_sizes)
    matching_detection_id = [
        idx for idx, v in enumerate(detection_sizes) if v == smallest_size
    ][0]
    matching_detection = detections[matching_detection_id]
    return (
        matching_detection["x"],
        matching_detection["y"],
        matching_detection["width"],
        matching_detection["height"],
    )


def get_largest_bounding_box(detections: List[dict]) -> Tuple[int, int, int, int]:
    detection_sizes = get_detection_sizes(detections=detections)
    largest_size = max(detection_sizes)
    matching_detection_id = [
        idx for idx, v in enumerate(detection_sizes) if v == largest_size
    ][0]
    matching_detection = detections[matching_detection_id]
    return (
        matching_detection["x"],
        matching_detection["y"],
        matching_detection[WIDTH_KEY],
        matching_detection[HEIGHT_KEY],
    )


AGGREGATION_MODE2BOXES_AGGREGATOR = {
    AggregationMode.MAX: get_largest_bounding_box,
    AggregationMode.MIN: get_smallest_bounding_box,
    AggregationMode.AVERAGE: get_average_bounding_box,
}


def get_detection_sizes(detections: List[dict]) -> List[float]:
    return [d[HEIGHT_KEY] * d[WIDTH_KEY] for d in detections]


AGGREGATION_MODE2FIELD_AGGREGATOR = {
    AggregationMode.MAX: max,
    AggregationMode.MIN: min,
    AggregationMode.AVERAGE: statistics.mean,
}


def aggregate_field_values(
    detections: List[dict],
    field: str,
    aggregation_mode: AggregationMode = AggregationMode.AVERAGE,
) -> float:
    values = [d[field] for d in detections]
    return AGGREGATION_MODE2FIELD_AGGREGATOR[aggregation_mode](values)
