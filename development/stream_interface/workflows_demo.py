import os
from functools import partial
from threading import Thread
from typing import List, Optional

import cv2
import supervision as sv

from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.camera.video_source import BufferFillingStrategy, BufferConsumptionStrategy
from inference.core.interfaces.stream.sinks import InMemoryBufferSink, WorkflowsStreamerSink, multi_sink
from inference.core.interfaces.stream.watchdog import PipelineWatchDog, BasePipelineWatchDog
from inference.core.utils.drawing import create_tiles

STOP = False
ANNOTATOR = sv.BoundingBoxAnnotator()
# TARGET_PROJECT = os.environ["TARGET_PROJECT"]
fps_monitor = sv.FPSMonitor()


def main() -> None:
    global STOP
    watchdog = BasePipelineWatchDog()
    workflow_specification = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
        ],
        "steps": [
            {
                "type": "ObjectDetectionModel",
                "name": "step_1",
                "image": "$inputs.image",
                "model_id": "yolov8n-640",
                "confidence": 0.5,
            },
            {
                "type": "roboflow_core/bounding_box_visualization@v1",
                "name": "bbox_visualiser",
                "predictions": "$steps.step_1.predictions",
                "image": "$inputs.image"
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "predictions", "selector": "$steps.step_1.predictions"},
            {"type": "JsonField", "name": "preview", "selector": "$steps.bbox_visualiser.image"},

        ],
    }
    buffer_sink = InMemoryBufferSink(
        queue_size=64,
    )
    streamer_sink = WorkflowsStreamerSink.init(
        pipeline_identifier="some",
        number_of_streams=2,
        image_outputs=["preview"],
        stream_server_url="127.0.0.1",
        rtsp_port=8554,
        webrtc_port=8889,
    )
    sinks = [buffer_sink.on_prediction, streamer_sink.on_prediction]
    sink = partial(multi_sink, sinks=sinks)
    pipeline = InferencePipeline.init_with_workflow(
        video_reference=["rtsp://localhost:8554/live.stream"],
        workflow_specification=workflow_specification,
        watchdog=watchdog,
        on_prediction=sink,
        source_buffer_filling_strategy=BufferFillingStrategy.DROP_OLDEST,
        source_buffer_consumption_strategy=BufferConsumptionStrategy.EAGER,
    )
    control_thread = Thread(target=command_thread, args=(pipeline, watchdog))
    control_thread.start()
    pipeline.start()
    STOP = True
    pipeline.join()


def command_thread(pipeline: InferencePipeline, watchdog: PipelineWatchDog) -> None:
    global STOP
    while not STOP:
        key = input()
        if key == "i":
            print(watchdog.get_report())
        if key == "t":
            pipeline.terminate()
            STOP = True
        elif key == "p":
            pipeline.pause_stream()
        elif key == "m":
            pipeline.mute_stream()
        elif key == "r":
            pipeline.resume_stream()


def workflows_sink(
    predictions: List[Optional[dict]],
    video_frames: List[Optional[VideoFrame]],
) -> None:
    fps_monitor.tick()
    if not isinstance(predictions, list):
        predictions = [predictions]
        video_frames = [video_frames]
    images_to_show = []
    for prediction, frame in zip(predictions, video_frames):
        if prediction is None or frame is None:
            continue
        detections: sv.Detections = prediction["predictions"]
        visualised = ANNOTATOR.annotate(frame.image.copy(), detections)
        images_to_show.append(visualised)
    tiles = create_tiles(images=images_to_show)
    cv2.imshow(f"Predictions", tiles)
    cv2.waitKey(1)
    if hasattr(fps_monitor, "fps"):
        fps_value = fps_monitor.fps
    else:
        fps_value = fps_monitor()
    print(f"FPS: {fps_value}")


if __name__ == '__main__':
    main()
