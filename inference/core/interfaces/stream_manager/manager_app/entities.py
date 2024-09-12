from enum import Enum
from typing import Union, List, Optional, Dict, Any

from pydantic import BaseModel

from inference.core.interfaces.camera.video_source import BufferFillingStrategy, BufferConsumptionStrategy

STATUS_KEY = "status"
TYPE_KEY = "type"
ERROR_TYPE_KEY = "error_type"
REQUEST_ID_KEY = "request_id"
PIPELINE_ID_KEY = "pipeline_id"
COMMAND_KEY = "command"
RESPONSE_KEY = "response"
ENCODING = "utf-8"


class OperationStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class ErrorType(str, Enum):
    INTERNAL_ERROR = "internal_error"
    INVALID_PAYLOAD = "invalid_payload"
    NOT_FOUND = "not_found"
    OPERATION_ERROR = "operation_error"
    AUTHORISATION_ERROR = "authorisation_error"


class CommandType(str, Enum):
    INIT = "init"
    MUTE = "mute"
    RESUME = "resume"
    STATUS = "status"
    TERMINATE = "terminate"
    LIST_PIPELINES = "list_pipelines"
    CONSUME_RESULT = "consume_result"


class SinkConfiguration(BaseModel):
    results_buffer_size: int = 64
    re_streaming_fields: Optional[List[str]] = None
    stream_server_url: Optional[str] = None
    server_rtsp_port: int = 8554
    server_webrtc_port: int = 8889


class InitialisePipelinePayload(BaseModel):
    video_reference: Union[str, int, List[Union[str, int]]]
    workflow_specification: Optional[dict] = None
    workspace_name: Optional[str] = None
    workflow_id: Optional[str] = None
    api_key: Optional[str] = None
    image_input_name: str = "image"
    workflows_parameters: Optional[Dict[str, Any]] = None
    sink_configuration: SinkConfiguration
    max_fps: Optional[Union[float, int]] = None
    source_buffer_filling_strategy: Optional[BufferFillingStrategy] = None
    source_buffer_consumption_strategy: Optional[BufferConsumptionStrategy] = None
    video_source_properties: Optional[Dict[str, float]] = None
    workflow_init_parameters: Optional[Dict[str, Any]] = None
    workflows_thread_pool_workers: int = 4
    cancel_thread_pool_tasks_on_exit: bool = True
    video_metadata_input_name: str = "video_metadata"


class ConsumeResultsPayload(BaseModel):
    purge: bool = False

