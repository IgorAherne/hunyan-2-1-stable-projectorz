# api_spz/core/models_pydantic.py


from enum import Enum
from typing import Optional, Dict
from pydantic import BaseModel, Field, ConfigDict, model_validator


class TaskStatus(str, Enum):
    PROCESSING = "PROCESSING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class GenerationArgForm(BaseModel):
    """A Pydantic model to validate the arguments from the client's JSON payload."""
    # Add model_config to ignore extra fields like 'generate_what' from the client
    model_config = ConfigDict(extra="ignore")

    seed: int = 1234
    guidance_scale: float = 5.0
    num_inference_steps: int = 20
    octree_resolution: int = 256
    num_chunks: int = 80
    unwrap_uv:bool = True,
    apply_texture: bool = False,
    mesh_simplify: float = 10.0
    texture_size: int = 4096 #final texture size (will upscale up to here). Rendering will still be at 768 etc.
    num_view_chunks: int = 3 # for controlling texture generation view chunking
    output_format: str = "glb"

    @model_validator(mode='before')
    @classmethod
    def cast_int_fields(cls, data):
        """Pre-processes incoming data to cast float-like ints to pure ints before validation."""
        int_fields = ["num_inference_steps", "octree_resolution", "num_chunks", "texture_size", "num_view_chunks", "seed"]
        for field in int_fields:
            if field in data and data[field] is not None:
                try:
                    data[field] = int(data[field])
                except (ValueError, TypeError):
                    pass # Let the standard Pydantic validation handle the error if casting fails
        return data



class GenerationResponse(BaseModel):
    status: TaskStatus
    progress: int = 0
    message: str = ""
    model_url: Optional[str] = None


class StatusResponse(BaseModel):
    status: TaskStatus
    progress: int
    message: str
    busy: bool