# hy3dpaint/hunyuanpaintpbr/unet/profiler.py

class BlockProfiler:
    _instance = None
    enabled = False
    data = {}

    @classmethod
    def enable(cls):
        cls.enabled = True
        cls.data = {
            "count": 0,
            "total_block_time": 0.0,
            "norm_time": 0.0,
            "self_attn_time": 0.0,
            "ref_attn_time": 0.0,
            "mv_attn_time": 0.0,
            "cross_attn_time": 0.0,
            "dino_attn_time": 0.0,
            "ff_time": 0.0,
            # Attention internals
            "qkv_proj_time": 0.0,
            "sdpa_time": 0.0,
            "rope_time": 0.0,
            "sdpa_time_count": 0,
        }

    @classmethod
    def disable(cls):
        cls.enabled = False

    @classmethod
    def reset(cls):
        cls.data = {}

    @classmethod
    def get_summary(cls):
        return cls.data