# Licensing

This repository's original runtime code is Apache-2.0. That grant does not relicense
models, upstream engines, custom nodes, their transitive dependencies, or the combined
container. Preserve upstream notices and source availability requirements when
publishing an image. This page is an inventory and operational reminder, not legal advice.

| Component | Observed licence / source |
| --- | --- |
| h3ltx original code | Apache-2.0, root LICENSE |
| MiniMax H3 weights, Qwen H3 conversion and turbo adapters | [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE); conversion/adapter grants do not remove base restrictions |
| LTX 2.5 weights and pinned Lightricks code | [LTX-2.x Community License](https://github.com/Lightricks/LTX-2/blob/e98f6d927d83c92ef2be6aec76c8440b19306e39/LICENSE-2_x) |
| Comfy-Org SeedVR2 weights | [Apache-2.0 model card](https://huggingface.co/Comfy-Org/SeedVR2) |
| LBH latent-upscaler weights | [Apache-2.0 model card](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) |
| ComfyUI | GPL-3.0; exact pinned source and licence retained in image |
| ComfyUI-VideoHelperSuite | GPL-3.0; exact pinned source and licence retained in image |
| LBH custom-node source | MIT; exact source and notice retained in image |
| Carasibana FaceRefine source | MIT; exact source and notice retained in image |
| Official workflow templates | MIT, copyright 2023-present Comfy Org; topology references, not vendored workflows |
| Bingsu face detector weights | [Apache-2.0 model card](https://huggingface.co/Bingsu/adetailer); optional pickle-format artifact |
| Ultralytics detector runtime | AGPL-3.0 or a separate commercial licence; review distribution/service obligations |
| SageAttention 1.0.6 | Apache-2.0 project metadata; separately locked dependency |
| NVIDIA CUDA/cuDNN, PyTorch, ffmpeg and other packages | Their own licences; inspect generated SBOM and bundled notices before distribution |

H3 excludes use in the United States, European Union, United Kingdom and South
Korea under its territory terms. Its attribution requirements include identifying
MiniMax H3 as required by the licence; retain applicable notices and display the
required attribution in downstream products/services. This API repository includes
that identification in its documentation; a downstream UI/service must assess its
own obligations. The doctor warning never makes a legal determination or blocks a
job. Iceland is outside the EU; independently confirm location and the current terms.

The LTX-2.x licence distinguishes commercial eligibility using a $10 million annual
revenue threshold. Organisations at or above that threshold need the applicable
separate permission/licence for commercial use. Read the complete definition,
restrictions and attribution terms rather than treating the threshold as the only
condition. Gated repository access does not itself establish commercial permission.

Native Gemma 4 projection weights are distributed in the gated LTX package; preserve
its model card and any upstream Gemma terms incorporated there. The manifest's
licence identifier names the distribution terms, not a blanket replacement of all
upstream rights. Optional detector and transitive packages also need review.

No third-party implementation source was copied into the Apache runtime. Its graph
builders and adapters were authored against node schemas and documented interfaces.
The image fetches immutable upstream source trees instead of rebranding them as
Apache code. `docker/sources.lock.json` records source commits and observed licences;
`models/manifest.yaml` records per-file model origins and terms.
