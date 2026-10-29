"""API graphs authored against official workflow topology and real node schemas."""

from typing import Any

from h3ltx.errors import RuntimeFault
from h3ltx.params import H3, LTX, Common, Face, ICLora, SeedVR
from h3ltx.schedules import distilled_sigmas
from h3ltx.workflows.validate import Graph

Link = list[Any]


class Builder:
    def __init__(self, files: dict[str, str] | None = None, prefix: str = "run"):
        self.graph: Graph = {}
        self.files = files or {}
        self.prefix = prefix

    def node(self, cls: str, **inputs: Any) -> Link:
        key = str(len(self.graph) + 1)
        self.graph[key] = {"class_type": cls, "inputs": inputs}
        return [key, 0]

    @staticmethod
    def slot(link: Link, slot: int) -> Link:
        return [link[0], slot]

    def file(self, identifier: str) -> str:
        # Validation builds can use deterministic placeholder names; runtime stages actual uploads.
        return self.files.get(identifier, identifier + ".png")

    def image(self, identifier: str) -> Link:
        return self.node("LoadImage", image=self.file(identifier))

    def video(self, identifier: str) -> Link:
        return self.node(
            "GetVideoComponents", video=self.node("LoadVideo", file=self.file(identifier))
        )

    def save(
        self, images: Link, audio: Link | None, fps: int | float | Link, hdr: bool = False
    ) -> None:
        inputs: dict[str, Any] = {
            "images": images,
            "fps": fps,
            "color_space": "HDR" if hdr else "sRGB",
            "bit_depth": "auto",
        }
        if audio:
            inputs["audio"] = audio
        video = self.node("CreateVideo", **inputs)
        self.node(
            "SaveVideo",
            video=video,
            filename_prefix=self.prefix,
            **{"format": "mp4", "format.codec": "h264"},
        )

    def sample(
        self,
        model: Link,
        positive: Link,
        negative: Link | None,
        latent: Link,
        seed: int,
        sampler: str = "euler",
        steps: int = 8,
        cfg: float = 1,
        sigmas: str | None = None,
        denoise: float = 1,
        scheduler: str = "simple",
    ) -> Link:
        guider = (
            self.node("CFGGuider", model=model, positive=positive, negative=negative, cfg=cfg)
            if negative
            else self.node("BasicGuider", model=model, conditioning=positive)
        )
        schedule = (
            self.node("ManualSigmas", sigmas=sigmas)
            if sigmas
            else self.node(
                "BasicScheduler", model=model, scheduler=scheduler, steps=steps, denoise=denoise
            )
        )
        return self.node(
            "SamplerCustomAdvanced",
            noise=self.node("RandomNoise", noise_seed=seed),
            guider=guider,
            sampler=self.node("KSamplerSelect", sampler_name=sampler),
            sigmas=schedule,
            latent_image=latent,
        )

    def decode(self, samples: Link, vae: Link, tiled: bool = True, tile_size: int = 512) -> Link:
        if tiled:
            return self.node(
                "VAEDecodeTiled",
                samples=samples,
                vae=vae,
                tile_size=tile_size,
                overlap=64,
                temporal_size=64,
                temporal_overlap=8,
            )
        return self.node("VAEDecode", samples=samples, vae=vae)

    def h3(self, kind: str, p: H3) -> Graph:
        reference = kind == "h3.r2v"
        variant = "ref2va" if reference else "fl2va"
        precision = {
            "pruned-bf16": "pruned_bf16",
            "full-bf16": "bf16",
            "int8": "pruned_int8_convrot",
            "nvfp4": "pruned_nvfp4",
        }[p.weights]
        model = self.node(
            "UNETLoader",
            unet_name=f"minimax_h3_{variant}_{precision}.safetensors",
            weight_dtype="default",
        )
        # The template relies on the model's default shift 12; explicit patch enables A/B tests.
        model = self.node("MiniMaxH3SigmaShift", model=model, shift_video=p.shift, shift_audio=3.0)
        steps = p.steps
        if p.turbo != "none":
            steps = 4 if p.turbo == "4-step" else 8
            suffix = "fl2v" if not reference else "ref2v"
            version = (
                ("v1.3_768p" if not reference and steps == 4 else "v1.1_768p")
                if p.turbo_version == "latest"
                else (
                    ("v0.1" if steps == 4 else "v1.1_768p")
                    if reference
                    else ("v1.0_768p" if steps == 4 else "v1.0")
                )
            )
            name = f"minimax_h3_{suffix}_turbo_{steps}step_{version}_comfyui_bf16.safetensors"
            model = self.node(
                "LoraLoaderModelOnly", model=model, lora_name=name, strength_model=1.0
            )
        clip = self.node(
            "CLIPLoader",
            clip_name="qwen3vl_32b_minimax_h3_bf16.safetensors",
            type="minimax",
            device="default",
        )
        vae = self.node("VAELoader", vae_name="minimax_h3_video_vae_fp16.safetensors")
        audio_vae = self.node("VAELoader", vae_name="minimax_h3_audio_vae_fp32.safetensors")
        from h3ltx.params import H3_FRAME_EXPRESSION

        length = self.slot(
            self.node(
                "ComfyMathExpression",
                expression=H3_FRAME_EXPRESSION,
                **{"values.a": p.duration_seconds},
            ),
            1,
        )
        conditioning: dict[str, Any] = dict(
            clip=clip, vae=vae, prompt=p.prompt, width=p.width, height=p.height, length=length
        )
        if reference:
            conditioning.update(audio_vae=audio_vae, ref_image_size="match")
            counts = {"image": 0, "video": 0, "audio": 0}
            for ref in p.references:
                i = counts[ref.kind]
                counts[ref.kind] += 1
                if ref.kind == "image":
                    conditioning[f"ref_images.ref_image_{i}"] = self.image(ref.upload_id)
                elif ref.kind == "video":
                    video = self.video(ref.upload_id)
                    conditioning[f"ref_videos.ref_video_{i}"] = video
                    conditioning[f"ref_video_audios.ref_video_audio_{i}"] = self.slot(video, 1)
                else:
                    conditioning[f"ref_audios.ref_audio_{i}"] = self.node(
                        "LoadAudio", audio=self.file(ref.upload_id)
                    )
            cond = self.node("MiniMaxH3ReferenceToVideo", **conditioning)
        else:
            if p.first_frame:
                conditioning["first_frame"] = self.image(p.first_frame)
            if p.last_frame:
                conditioning["last_frame"] = self.image(p.last_frame)
            cond = self.node("MiniMaxH3ImageToVideo", **conditioning)
        positive, latent = cond, self.slot(cond, 1)
        for ref in p.guides:
            extra: dict[str, Any] = {}
            if ref.kind == "image":
                extra["image"] = self.image(ref.upload_id)
            elif ref.kind == "video":
                components = self.video(ref.upload_id)
                extra.update(image=components, audio=self.slot(components, 1))
            else:
                extra["audio"] = self.node("LoadAudio", audio=self.file(ref.upload_id))
            positive = self.node(
                "MiniMaxH3AddGuide",
                positive=positive,
                latent=latent,
                vae=vae,
                audio_vae=audio_vae,
                frame_idx=ref.frame,
                **extra,
            )
        sampled = self.sample(
            model, positive, None, latent, p.seed, p.sampler, steps, scheduler=p.scheduler
        )
        if p.latent_upscale == "2x":
            upscaler = self.node(
                "LatentUpscaleModelLoader",
                model_name="minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
            )
            latent = self.node(
                "MiniMaxH3LatentUpscaleWithModel",
                model=upscaler,
                samples=sampled,
                width=p.width * 2,
                height=p.height * 2,
            )
            sampled = self.sample(
                model,
                positive,
                None,
                latent,
                (p.seed + 1) % 2**64,
                p.sampler,
                p.refine_steps,
                denoise=0.25,
                scheduler=p.scheduler,
            )
        images = self.decode(sampled, vae, tiled=False)
        audio = self.node("VAEDecodeAudio", samples=sampled, vae=audio_vae)
        self.save(images, audio, 24)
        return self.graph

    def ltx(self, p: LTX) -> Graph:
        dfr = p.pipeline == "dfr"
        model_kind = "dev" if p.pipeline == "dev" else "distilled"
        precision = {"bf16": "bf16", "int8": "comfy-int8-convrot", "nvfp4": "nvfp4"}[p.weights]
        model = self.node(
            "UNETLoader",
            unet_name=f"ltx-2.5-22b-{model_kind}-transformer-{precision}.safetensors",
            weight_dtype="default",
        )
        clip = self.node(
            "CLIPLoader",
            clip_name="gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
            type="ltxv",
            device="default",
        )
        vae = self.node("VAELoader", vae_name="ltx-2.5-video-vae-bf16.safetensors")
        audio_vae = self.node("VAELoader", vae_name="ltx-2.5-audio-vae-bf16.safetensors")
        positive = self.node("CLIPTextEncode", clip=clip, text=p.prompt)
        negative = self.node("CLIPTextEncode", clip=clip, text=p.negative_prompt)
        conditioning = self.node(
            "LTXVConditioning", positive=positive, negative=negative, frame_rate=p.fps
        )
        positive, negative = conditioning, self.slot(conditioning, 1)
        latent = self.node(
            "EmptyLTXVLatentVideo",
            width=p.width // 2,
            height=p.height // 2,
            length=p.frames,
            batch_size=1,
        )
        for ref in p.keyframes:
            image = self.image(ref.upload_id)
            if ref.frame == 0:
                latent = self.node(
                    "LTXVImgToVideoInplace",
                    vae=vae,
                    image=image,
                    latent=latent,
                    strength=ref.strength,
                    bypass=False,
                )
            else:
                guide = self.node(
                    "LTXVAddGuide",
                    positive=positive,
                    negative=negative,
                    vae=vae,
                    latent=latent,
                    image=image,
                    frame_idx=ref.frame,
                    strength=ref.strength,
                )
                positive, negative, latent = guide, self.slot(guide, 1), self.slot(guide, 2)
        if dfr:
            generated = self.node(
                "LTXVAddGeneratedKeyframes",
                positive=positive,
                negative=negative,
                vae=vae,
                latent=latent,
                interval_frames=24,
            )
            positive, negative, latent = generated, self.slot(generated, 1), self.slot(generated, 2)
        audio_latent = self.node(
            "LTXVEmptyLatentAudio",
            audio_vae=audio_vae,
            frames_number=p.frames,
            frame_rate=p.fps,
            batch_size=1,
        )
        av = self.node("LTXVConcatAVLatent", video_latent=latent, audio_latent=audio_latent)
        first_sigmas = ",".join(str(v) for v in distilled_sigmas(p.steps))
        sampled = self.sample(
            model,
            positive,
            negative,
            av,
            p.seed,
            p.sampler,
            p.steps,
            p.cfg,
            sigmas=first_sigmas if p.pipeline != "dev" else None,
        )
        separated = self.node("LTXVSeparateAVLatent", av_latent=sampled)
        video, audio_latent = separated, self.slot(separated, 1)
        keyframes: Link | None = None
        if dfr:
            peeled = self.node(
                "LTXVSeparateGeneratedKeyframes",
                positive=positive,
                negative=negative,
                latent=video,
                keyframes_to_batch=False,
            )
            positive, negative, video, keyframes = (
                peeled,
                self.slot(peeled, 1),
                self.slot(peeled, 2),
                self.slot(peeled, 3),
            )
        low_video = video
        spatial = self.node(
            "LatentUpscaleModelLoader",
            model_name="ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        )
        video = self.node("LTXVLatentUpsampler", samples=video, upscale_model=spatial, vae=vae)
        if dfr:
            model = self.node(
                "LoraLoaderModelOnly",
                model=model,
                lora_name="ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors",
                strength_model=1.0,
            )
            assert keyframes is not None
            keyframes = self.node(
                "LTXVLatentUpsampler", samples=keyframes, upscale_model=spatial, vae=vae
            )
            generated = self.node(
                "LTXVAddGeneratedKeyframes",
                positive=positive,
                negative=negative,
                latent=video,
                vae=vae,
                keyframes=keyframes,
                interval_frames=24,
            )
            positive, negative, video = generated, self.slot(generated, 1), self.slot(generated, 2)
            guided = self.node(
                "LTXVAddLatentGuide",
                positive=positive,
                negative=negative,
                vae=vae,
                latent=video,
                guiding_latent=low_video,
                latent_idx=0,
                strength=1.0,
            )
            positive, negative, video = guided, self.slot(guided, 1), self.slot(guided, 2)
        elif p.pipeline == "dev":
            model = self.node(
                "LoraLoaderModelOnly",
                model=model,
                lora_name="ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
                strength_model=1.0,
            )
        av = self.node("LTXVConcatAVLatent", video_latent=video, audio_latent=audio_latent)
        sampled = self.sample(
            model,
            positive,
            negative,
            av,
            (p.seed + 1) % 2**64,
            p.sampler,
            cfg=p.cfg,
            sigmas="0.909375,0.725,0.421875,0.0" if dfr else "0.85,0.725,0.4219,0.0",
        )
        separated = self.node("LTXVSeparateAVLatent", av_latent=sampled)
        video, audio_latent = separated, self.slot(separated, 1)
        if dfr:
            peeled = self.node(
                "LTXVSeparateGeneratedKeyframes",
                positive=positive,
                negative=negative,
                latent=video,
                keyframes_to_batch=False,
            )
            positive, negative, video, keyframes = (
                peeled,
                self.slot(peeled, 1),
                self.slot(peeled, 2),
                self.slot(peeled, 3),
            )
            cropped = self.node(
                "LTXVCropGuides", positive=positive, negative=negative, latent=video
            )
            positive, negative, video = cropped, self.slot(cropped, 1), self.slot(cropped, 2)
        fps = p.fps
        for iteration in range(p.temporal_upscale):
            fps *= 2
            temporal = self.node(
                "LatentUpscaleModelLoader",
                model_name="ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
            )
            video = self.node("LTXVLatentUpsampler", samples=video, upscale_model=temporal, vae=vae)
            stamped = self.node(
                "LTXVConditioning", positive=positive, negative=negative, frame_rate=fps
            )
            positive, negative = stamped, self.slot(stamped, 1)
            if keyframes:
                guide = self.node(
                    "LTXVGeneratedKeyframesToGuides",
                    positive=positive,
                    negative=negative,
                    vae=vae,
                    latent=video,
                    keyframes=keyframes,
                    strength=0.95,
                )
                positive, negative, video = guide, self.slot(guide, 1), self.slot(guide, 2)
                generated = self.node(
                    "LTXVAddGeneratedKeyframes",
                    positive=positive,
                    negative=negative,
                    vae=vae,
                    latent=video,
                    keyframes=keyframes,
                    interval_frames=24,
                )
                positive, negative, video = (
                    generated,
                    self.slot(generated, 1),
                    self.slot(generated, 2),
                )
            frozen = self.node("LTXVFreezeLatent", latent=audio_latent)
            av = self.node("LTXVConcatAVLatent", video_latent=video, audio_latent=frozen)
            sampled = self.sample(
                model,
                positive,
                negative,
                av,
                (p.seed + iteration + 2) % 2**64,
                p.sampler,
                cfg=p.cfg,
                sigmas="0.975,0.909375,0.725,0.421875,0.0",
            )
            separated = self.node("LTXVSeparateAVLatent", av_latent=sampled)
            video = separated
            if keyframes:
                peeled = self.node(
                    "LTXVSeparateGeneratedKeyframes",
                    positive=positive,
                    negative=negative,
                    latent=video,
                    keyframes_to_batch=False,
                )
                positive, negative, video, keyframes = (
                    peeled,
                    self.slot(peeled, 1),
                    self.slot(peeled, 2),
                    self.slot(peeled, 3),
                )
            cropped = self.node(
                "LTXVCropGuides", positive=positive, negative=negative, latent=video
            )
            positive, negative, video = cropped, self.slot(cropped, 1), self.slot(cropped, 2)
        if p.hdr:
            # Dedicated IC-LoRA refinement on the full-resolution latent.
            hdr_model = self.node(
                "LoraLoaderModelOnly",
                model=model,
                lora_name="ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors",
                strength_model=1.0,
            )
            av = self.node(
                "LTXVConcatAVLatent",
                video_latent=video,
                audio_latent=self.node("LTXVFreezeLatent", latent=audio_latent),
            )
            sampled = self.sample(
                hdr_model,
                positive,
                negative,
                av,
                (p.seed + 100) % 2**64,
                sigmas="0.909375,0.725,0.421875,0.0",
            )
            video = self.node("LTXVSeparateAVLatent", av_latent=sampled)
        images = self.decode(video, vae, p.tiled)
        audio = self.node("LTXVAudioVAEDecode", samples=audio_latent, audio_vae=audio_vae)
        self.save(images, audio, fps, p.hdr)
        return self.graph

    def seedvr(self, p: SeedVR) -> Graph:
        components = self.video(p.input_video)
        factor = p.target_short_side / min(p.width, p.height)
        images = self.node(
            "ImageScale",
            image=components,
            upscale_method="lanczos",
            width=round(p.width * factor),
            height=round(p.height * factor),
            crop="disabled",
        )
        clean = self.node("SeedVR2Preprocess", resized_images=images)
        vae = self.node("VAELoader", vae_name="seedvr2_ema_vae_fp16.safetensors")
        variant = {"7b": "7b", "7b-sharp": "7b_sharp", "3b": "3b"}[p.model]
        model = self.node(
            "UNETLoader", unet_name=f"seedvr2_{variant}_fp16.safetensors", weight_dtype="default"
        )
        if p.tiled:
            latent = self.node(
                "VAEEncodeTiled",
                pixels=clean,
                vae=vae,
                tile_size=p.tile_size,
                overlap=64,
                temporal_size=64,
                temporal_overlap=8,
            )
        else:
            latent = self.node("VAEEncode", pixels=clean, vae=vae)
        chunked = self.node(
            "SeedVR2TemporalChunk",
            latent=latent,
            temporal_overlap=0,
            **{"chunking_mode": "manual", "chunking_mode.frames_per_chunk": p.frames_per_chunk},
        )
        cond = self.node("SeedVR2Conditioning", model=model, vae_conditioning=chunked)
        sampled = self.sample(
            model, cond, self.slot(cond, 1), chunked, p.seed, "euler", 1, 1, sigmas="1.0,0.0"
        )
        merged = self.node(
            "SeedVR2TemporalMerge", latents=sampled, temporal_overlap=self.slot(chunked, 1)
        )
        decoded = self.decode(merged, vae, p.tiled, p.tile_size)
        final = self.node(
            "SeedVR2PostProcessing",
            images=decoded,
            original_resized_images=images,
            color_correction_method="lab",
        )
        self.save(final, self.slot(components, 1), self.slot(components, 2))
        return self.graph

    def iclora(self, p: ICLora) -> Graph:
        components = self.video(p.input_video)
        vae = self.node("VAELoader", vae_name="ltx-2.5-video-vae-bf16.safetensors")
        clip = self.node(
            "CLIPLoader",
            clip_name="gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
            type="ltxv",
            device="default",
        )
        model = self.node(
            "UNETLoader",
            unet_name="ltx-2.5-22b-distilled-transformer-bf16.safetensors",
            weight_dtype="default",
        )
        lora = "pixel-spatial-upscaler-x2" if p.mode == "pixel-spatial" else "refine-details"
        model = self.node(
            "LoraLoaderModelOnly",
            model=model,
            lora_name=f"ltx-2.5-22b-ic-lora-{lora}-1.0.safetensors",
            strength_model=1.0,
        )
        positive = self.node("CLIPTextEncode", clip=clip, text=p.prompt)
        negative = self.node("CLIPTextEncode", clip=clip, text="")
        stamped = self.node(
            "LTXVConditioning", positive=positive, negative=negative, frame_rate=p.fps
        )
        positive, negative = stamped, self.slot(stamped, 1)
        encoded = self.node("VAEEncode", pixels=components, vae=vae)
        if p.mode == "pixel-spatial":
            spatial = self.node(
                "LatentUpscaleModelLoader",
                model_name="ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
            )
            video = self.node(
                "LTXVLatentUpsampler", samples=encoded, upscale_model=spatial, vae=vae
            )
        else:
            video = encoded
        guide = self.node(
            "LTXVAddLatentGuide",
            positive=positive,
            negative=negative,
            vae=vae,
            latent=video,
            guiding_latent=encoded,
            latent_idx=0,
            strength=1.0,
        )
        sampled = self.sample(
            model,
            guide,
            self.slot(guide, 1),
            self.slot(guide, 2),
            p.seed,
            sigmas="0.909375,0.725,0.421875,0.0",
        )
        cropped = self.node(
            "LTXVCropGuides", positive=guide, negative=self.slot(guide, 1), latent=sampled
        )
        self.save(self.decode(self.slot(cropped, 2), vae, p.tiled), self.slot(components, 1), p.fps)
        return self.graph

    def face(self, p: Face) -> Graph:
        components = self.video(p.input_video)
        crop = self.node(
            "H3FaceTrackCrop",
            images=components,
            detector=p.detector,
            confidence=0.35,
            crop_factor=2.5,
            canvas_width=768,
            canvas_height=768,
            canvas_mode="manual",
            smooth_window=21,
            size_smooth_window=51,
            smooth_method="gaussian",
            size_mode="per_frame",
        )
        vae = self.node("VAELoader", vae_name="minimax_h3_video_vae_fp16.safetensors")
        clip = self.node(
            "CLIPLoader",
            clip_name="qwen3vl_32b_minimax_h3_bf16.safetensors",
            type="minimax",
            device="default",
        )
        model = self.node(
            "UNETLoader",
            unet_name="minimax_h3_ref2va_pruned_bf16.safetensors",
            weight_dtype="default",
        )
        from h3ltx.params import h3_frames

        first_crop = self.node("ImageFromBatch", image=crop, batch_index=0, length=1)
        cond = self.node(
            "MiniMaxH3ReferenceToVideo",
            clip=clip,
            vae=vae,
            prompt=p.prompt,
            width=768,
            height=768,
            length=h3_frames(p.frames / 24),
            ref_image_size="match",
            **{"ref_images.ref_image_0": first_crop},
        )
        injected = self.node(
            "H3InjectVideoLatent", av_latent=self.slot(cond, 1), images=crop, vae=vae
        )
        adaptive = self.node(
            "H3PerFrameDenoise",
            model=model,
            av_latent=injected,
            transform=self.slot(crop, 1),
            denoise_multiplier_small_face=1.0,
            denoise_multiplier_large_face=0.35,
            scale_mode="absolute_px",
            face_px_small=30.0,
            face_px_large=120.0,
            gamma=1.0,
            smooth_frames=9,
        )
        sampled = self.sample(
            self.slot(adaptive, 2),
            cond,
            None,
            adaptive,
            p.seed,
            "res_multistep",
            p.steps,
            denoise=p.denoise,
        )
        decoded = self.decode(sampled, vae, tiled=False)
        final = self.node(
            "H3FaceStitch",
            base_images=components,
            refined_crops=decoded,
            transform=self.slot(crop, 1),
            paste_region="face_only",
            mask_dilation=16,
            feather=6,
            colour_match=1.0,
            blend=1.0,
            undetected_frames="fade_out",
        )
        self.save(final, self.slot(components, 1), self.slot(components, 2))
        return self.graph


def build(
    kind: str, params: Common, files: dict[str, str] | None = None, prefix: str = "run"
) -> Graph:
    b = Builder(files, prefix)
    if isinstance(params, H3):
        return b.h3(kind, params)
    if isinstance(params, LTX):
        return b.ltx(params)
    if isinstance(params, SeedVR):
        return b.seedvr(params)
    if isinstance(params, ICLora):
        return b.iclora(params)
    if isinstance(params, Face):
        return b.face(params)
    raise RuntimeFault("builder_missing", f"No graph builder for {kind}")
