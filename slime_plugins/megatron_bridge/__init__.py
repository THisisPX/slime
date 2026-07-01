try:
    import slime_plugins.megatron_bridge.glm4v_moe  # noqa: F401  # register GLM-4.6V bridge
except ImportError:
    pass  # GLM-4.6V bridge unavailable (e.g. incompatible megatron-bridge version)
