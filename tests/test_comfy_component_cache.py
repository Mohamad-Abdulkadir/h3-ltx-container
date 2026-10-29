"""Exercise the actual cache extension against fake ComfyUI model-management objects."""

import sys
from types import ModuleType, SimpleNamespace

from h3ltx import comfy_extension as extension


def test_loader_objects_survive_new_prompts_and_transfer_without_disk_reload(monkeypatch):
    class Patcher:
        load_device = "cuda"

        def __init__(self):
            self.used = 0

        def model_size(self):
            return 10

        def loaded_size(self):
            return self.used

    class V3:
        pass

    reads = {}
    models = []

    def legacy(label):
        def load(self, **inputs):
            reads[label] = reads.get(label, 0) + 1
            patcher = Patcher()
            models.append(patcher)
            return (
                SimpleNamespace(patcher=patcher)
                if label in ("CLIPLoader", "VAELoader")
                else patcher,
            )

        return type(label, (), {"FUNCTION": "load", "load": load})

    classes = {
        label: legacy(label)
        for label in (
            "UNETLoader",
            "CLIPLoader",
            "VAELoader",
            "LoraLoaderModelOnly",
            "LatentUpscaleModelLoader",
            "MiniMaxH3SigmaShift",
        )
    }

    class Sampler(V3):
        @classmethod
        def execute(cls, **inputs):
            return inputs["latent_image"]

    classes["SamplerCustomAdvanced"] = Sampler
    memory = ModuleType("comfy.model_management")
    memory.current_loaded_models = []

    def load_gpu(patchers, **kwargs):
        for patcher in patchers:
            patcher.used = 10
            if not any(item.model is patcher for item in memory.current_loaded_models):
                memory.current_loaded_models.append(SimpleNamespace(model=patcher))

    def free(*args, keep_loaded):
        for model in memory.current_loaded_models:
            if model not in keep_loaded:
                model.model.used = 0

    memory.load_models_gpu = load_gpu
    memory.free_memory = free
    memory.get_torch_device = lambda: "cuda"
    memory.soft_empty_cache = lambda: None
    memory.unload_all_models = lambda: free(keep_loaded=[])
    root = ModuleType("comfy")
    root.model_management = memory
    nodes = ModuleType("nodes")
    nodes.NODE_CLASS_MAPPINGS = classes
    latest = ModuleType("comfy_api.latest")
    latest.io = SimpleNamespace(ComfyNode=V3)
    api = ModuleType("comfy_api")
    api.latest = latest
    aiohttp = ModuleType("aiohttp")
    aiohttp.web = SimpleNamespace(json_response=lambda value: value)
    server = ModuleType("server")
    server.PromptServer = SimpleNamespace(
        instance=SimpleNamespace(routes=SimpleNamespace(get=lambda route: lambda fn: fn))
    )
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True, synchronize=lambda: None)
    for name, module in {
        "comfy": root,
        "comfy.model_management": memory,
        "nodes": nodes,
        "comfy_api": api,
        "comfy_api.latest": latest,
        "aiohttp": aiohttp,
        "server": server,
        "torch": torch,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(extension, "_CACHE", {})
    monkeypatch.setattr(extension, "_PIPELINES", {})
    monkeypatch.setattr(extension, "_INSTALLED", False)
    extension.install()
    model = classes["UNETLoader"]().load(unet_name="model.safetensors")[0]
    clip = classes["CLIPLoader"]().load(clip_name="encoder.safetensors")[0]
    vae = classes["VAELoader"]().load(vae_name="vae.safetensors")[0]
    report = extension.H3LTXPreload().operate(
        "load",
        "h3-fl2v",
        "first",
        '["dit","clip","vae"]',
        component_0=model,
        component_1=clip,
        component_2=vae,
    )
    assert report["ui"]["h3ltx_load"][0]["resident"]["h3-fl2v"]["resident"]
    # A new graph and fresh node instances have identical loader inputs. This is independent
    # of execution_cached and remains true with ComfyUI --cache-none.
    assert classes["UNETLoader"]().load(unet_name="model.safetensors")[0] is model
    assert classes["CLIPLoader"]().load(clip_name="encoder.safetensors")[0] is clip
    assert classes["VAELoader"]().load(vae_name="vae.safetensors")[0] is vae
    assert reads == {"UNETLoader": 1, "CLIPLoader": 1, "VAELoader": 1}
    monkeypatch.setenv("H3LTX_TEXT_OFFLOAD", "auto")
    assert Sampler.execute(latent_image="sample") == "sample"
    assert clip.patcher.used == 0 and model.used == 10
    assert not extension.residency()["h3-fl2v"]["resident"]
    extension.H3LTXPreload().operate("unload", "all", "second", "[]")
    assert not extension._CACHE and not extension._PIPELINES
    assert model.used == 0
