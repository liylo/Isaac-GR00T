# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import math
from pathlib import Path
from typing import Any, Tuple

import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.dit import AlternateVLDiT, DiT, SelfAttentionTransformer
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)


logger = logging.getLogger(__name__)


class Gr00tTaskStatusHead(nn.Module):
    """Attention-pool VLM tokens into completion and progress logits."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Task-status embedding and hidden dimensions must be positive")
        if num_heads <= 0 or embedding_dim % num_heads != 0:
            raise ValueError(
                "task_status_num_heads must divide backbone_embedding_dim, got "
                f"{num_heads} and {embedding_dim}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"task_status_dropout must be in [0, 1), got {dropout}")

        self.query = nn.Parameter(torch.empty(1, 1, embedding_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize every status-head parameter, including missing checkpoint weights.

        ``PreTrainedModel.from_pretrained`` may construct the model with torch's
        initialization functions disabled. The pipeline therefore calls this method
        again after it confirms that the whole head is absent from an older checkpoint.
        """
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.attention.in_proj_weight)
        if self.attention.in_proj_bias is not None:
            nn.init.zeros_(self.attention.in_proj_bias)
        nn.init.xavier_uniform_(self.attention.out_proj.weight)
        if self.attention.out_proj.bias is not None:
            nn.init.zeros_(self.attention.out_proj.bias)
        self.norm.reset_parameters()
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        backbone_features: torch.Tensor,
        backbone_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if backbone_features.ndim != 3:
            raise ValueError(
                "backbone_features must have shape [B, sequence, embedding], got "
                f"{tuple(backbone_features.shape)}"
            )

        key_padding_mask = None
        if backbone_attention_mask is not None:
            if backbone_attention_mask.shape != backbone_features.shape[:2]:
                raise ValueError(
                    "backbone_attention_mask must match the first two feature dimensions, got "
                    f"{tuple(backbone_attention_mask.shape)} vs "
                    f"{tuple(backbone_features.shape[:2])}"
                )
            valid_tokens = backbone_attention_mask.bool()
            if not torch.all(valid_tokens.any(dim=1)):
                raise ValueError("Every task-status sample must contain at least one valid token")
            key_padding_mask = ~valid_tokens

            # Masked backbone positions are not guaranteed to contain finite values.
            # Replacing them before attention prevents NaNs from leaking through a
            # subsequent additive attention mask.
            backbone_features = backbone_features.masked_fill(
                key_padding_mask.unsqueeze(-1),
                0,
            )

        # The backbone is trained under BF16 autocast. Keep the small trainable
        # status head in its parameter dtype (FP32 for finetuning) because attention
        # reductions are otherwise the first place non-finite logits can appear.
        with torch.autocast(device_type=backbone_features.device.type, enabled=False):
            backbone_features = backbone_features.to(dtype=self.query.dtype)
            query = self.query.expand(backbone_features.shape[0], -1, -1)
            pooled, _ = self.attention(
                query=query,
                key=backbone_features,
                value=backbone_features,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            logits = self.mlp(self.norm(pooled[:, 0]))
        if not torch.isfinite(logits).all():
            raise FloatingPointError(
                "Task-status head produced non-finite logits; check the source checkpoint "
                "and backbone features"
            )
        return logits


class Gr00tN1d7ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d7Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim
        if config.action_branch_mode not in {"primary", "eef", "primary_eef"}:
            raise ValueError(
                "action_branch_mode must be primary, eef, or primary_eef; got "
                f"{config.action_branch_mode!r}"
            )

        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            logger.info("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )
            logger.info("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        # EEF is an auxiliary task, not another embodiment. Give it independent
        # flow input/output adapters while retaining the shared state/context and
        # DiT trunk. One category is sufficient for both human and robot EEF.
        self.eef_action_encoder = None
        self.eef_action_decoder = None
        if config.action_branch_mode in {"eef", "primary_eef"}:
            self.eef_action_encoder = MultiEmbodimentActionEncoder(
                action_dim=self.action_dim,
                hidden_size=self.input_embedding_dim,
                num_embodiments=1,
            )
            self.eef_action_decoder = CategorySpecificMLP(
                num_categories=1,
                input_dim=self.hidden_size,
                hidden_dim=self.hidden_size,
                output_dim=self.action_dim,
            )
        if (
            not math.isfinite(config.auxiliary_action_loss_weight)
            or config.auxiliary_action_loss_weight <= 0
        ):
            raise ValueError(
                "auxiliary_action_loss_weight must be finite and positive, got "
                f"{config.auxiliary_action_loss_weight}"
            )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        vl_self_attention_cfg = getattr(config, "vl_self_attention_cfg", None)
        if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0:
            self.vl_self_attention = SelfAttentionTransformer(**vl_self_attention_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.eef_action_encoder is not None:
                self.eef_action_encoder.requires_grad_(False)
                self.eef_action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
            self.vl_self_attention.requires_grad_(False)
        if self.config.action_branch_mode == "eef":
            # Human EEF-only training must not touch token-specific adapters.
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
        logger.debug(f"Tune action head projector: {self.tune_projector}")
        logger.debug(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        logger.debug(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, log a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    logger.debug(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.eef_action_encoder is not None:
                    self.eef_action_encoder.eval()
                    self.eef_action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
            if not self.tune_vlln:
                self.vlln.eval()
                self.vl_self_attention.eval()
            if self.config.action_branch_mode == "eef":
                self.action_encoder.eval()
                self.action_decoder.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    @torch.no_grad()
    def initialize_eef_branch_from_primary(self, category_id: int | None = None) -> None:
        """Initialize a missing EEF branch from one pretrained primary category."""
        if self.eef_action_encoder is None or self.eef_action_decoder is None:
            raise RuntimeError("Cannot initialize EEF branch in primary-only mode")
        category_id = (
            self.config.auxiliary_action_init_category if category_id is None else category_id
        )
        if not 0 <= category_id < self.config.max_num_embodiments:
            raise ValueError(
                f"EEF initialization category {category_id} is outside "
                f"[0, {self.config.max_num_embodiments})"
            )

        def copy_category(target, source) -> None:
            target.W[0].copy_(source.W[category_id])
            target.b[0].copy_(source.b[category_id])

        for target, source in (
            (self.eef_action_encoder.W1, self.action_encoder.W1),
            (self.eef_action_encoder.W2, self.action_encoder.W2),
            (self.eef_action_encoder.W3, self.action_encoder.W3),
            (self.eef_action_decoder.layer1, self.action_decoder.layer1),
            (self.eef_action_decoder.layer2, self.action_decoder.layer2),
        ):
            copy_category(target, source)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def _run_flow_branch(
        self,
        *,
        actions: torch.Tensor,
        loss_mask: torch.Tensor,
        state_features: torch.Tensor,
        backbone_output: BatchFeature,
        action_encoder: MultiEmbodimentActionEncoder,
        action_decoder: CategorySpecificMLP,
        adapter_ids: torch.Tensor,
        mask_corruption: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one task-specific flow branch through the shared DiT trunk."""
        if mask_corruption:
            active_dims = (loss_mask > 0).to(dtype=actions.dtype)
            branch_actions = actions * active_dims
            noise = torch.randn_like(actions) * active_dims
        else:
            # Preserve legacy behavior for configurations without an auxiliary task.
            branch_actions = actions
            noise = torch.randn_like(actions)

        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * branch_actions
        velocity = branch_actions - noise
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = action_encoder(noisy_trajectory, t_discretized, adapter_ids)

        if self.config.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=actions.device
            )
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_embeds = backbone_output.backbone_features
        if self.config.use_alternate_vl_dit:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=backbone_output.backbone_attention_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=backbone_output.backbone_attention_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        pred = action_decoder(model_output, adapter_ids)
        pred_actions = pred[:, -actions.shape[1] :]
        elementwise_loss = F.mse_loss(pred_actions, velocity, reduction="none") * loss_mask
        normalized_loss = elementwise_loss.sum() / (loss_mask.sum() + 1e-6)
        return elementwise_loss, normalized_loss

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Handle state history
        assert action_input.state.shape[1] == self.config.state_history_length
        action_input.state = action_input.state.view(action_input.state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Dropout state features (training only): zero out dropped states.
        if self.training and self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout)

        actions = action_input.action
        action_mask = action_input.action_mask
        auxiliary_selector = action_input.get("auxiliary_action_mask")
        if auxiliary_selector is None:
            auxiliary_selector = torch.zeros_like(action_mask)
        if auxiliary_selector.shape != action_mask.shape:
            raise ValueError(
                "auxiliary_action_mask shape must match action_mask, got "
                f"{tuple(auxiliary_selector.shape)} vs {tuple(action_mask.shape)}"
            )
        auxiliary_selector = (auxiliary_selector > 0).to(dtype=action_mask.dtype)
        auxiliary_mask = action_mask * auxiliary_selector
        primary_mask = action_mask * (1 - auxiliary_selector)
        has_auxiliary = bool(torch.count_nonzero(auxiliary_mask).item())
        has_primary = bool(torch.count_nonzero(primary_mask).item())
        if not has_primary and not has_auxiliary:
            raise ValueError("Action batch contains no supervised primary or auxiliary dimensions")
        if self.config.action_branch_mode == "eef" and has_primary:
            raise ValueError("EEF-only branch mode received primary action supervision")
        if self.config.action_branch_mode == "primary" and has_auxiliary:
            raise ValueError("Primary-only branch mode received auxiliary EEF supervision")

        zero_loss = actions.new_zeros(())
        primary_action_loss = torch.zeros_like(actions)
        auxiliary_action_loss = torch.zeros_like(actions)
        primary_loss = zero_loss
        auxiliary_loss = zero_loss

        if has_primary:
            primary_action_loss, primary_loss = self._run_flow_branch(
                actions=actions,
                loss_mask=primary_mask,
                state_features=state_features,
                backbone_output=backbone_output,
                action_encoder=self.action_encoder,
                action_decoder=self.action_decoder,
                adapter_ids=embodiment_id,
                # Combined checkpoints always train the primary flow with
                # auxiliary lanes fixed at zero, including primary-only batches.
                mask_corruption=self.config.action_branch_mode == "primary_eef",
            )
        if has_auxiliary:
            if self.eef_action_encoder is None or self.eef_action_decoder is None:
                raise RuntimeError(
                    "Auxiliary EEF labels require action_branch_mode 'eef' or 'primary_eef'"
                )
            auxiliary_ids = torch.zeros_like(embodiment_id)
            auxiliary_action_loss, auxiliary_loss = self._run_flow_branch(
                actions=actions,
                loss_mask=auxiliary_mask,
                state_features=state_features,
                backbone_output=backbone_output,
                action_encoder=self.eef_action_encoder,
                action_decoder=self.eef_action_decoder,
                adapter_ids=auxiliary_ids,
                mask_corruption=True,
            )

        if has_primary and has_auxiliary:
            loss = primary_loss + self.config.auxiliary_action_loss_weight * auxiliary_loss
        elif has_primary:
            loss = primary_loss
        else:
            # EEF-only human pretraining keeps a unit-normalized objective. The
            # auxiliary weight only controls competition with the main task.
            loss = auxiliary_loss
        action_loss = primary_action_loss + auxiliary_action_loss

        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "primary_loss": primary_loss,
            "auxiliary_loss": auxiliary_loss,
            "token_loss": primary_loss,
            "eef_loss": auxiliary_loss,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_history_length, max_state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, 1, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Handle state history: if we have fewer timesteps than expected, repeat to fill
        state = action_input.state
        current_T = state.shape[1]
        assert current_T == self.config.state_history_length, "current_T != state_history_length"
        # Reshape state from [B, state_history_length, max_state_dim] to [B, 1, state_history_length * max_state_dim]
        state = state.view(state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features

        if self.config.action_branch_mode == "eef":
            raise RuntimeError("EEF-only checkpoints provide training supervision, not actions")

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        primary_action_mask = torch.ones(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )
        if self.config.action_branch_mode == "primary_eef":
            provided_mask = action_input.get("primary_action_mask")
            if provided_mask is None:
                raise RuntimeError(
                    "primary_eef inference requires primary_action_mask from the processor"
                )
            if provided_mask.shape != primary_action_mask.shape:
                raise ValueError(
                    "primary_action_mask shape must match sampled actions, got "
                    f"{tuple(provided_mask.shape)} vs {tuple(primary_action_mask.shape)}"
                )
            primary_action_mask = (provided_mask > 0).to(device=device, dtype=vl_embeds.dtype)
            if not torch.count_nonzero(primary_action_mask).item():
                raise RuntimeError("primary_eef inference mask contains no primary action slots")
        actions = torch.randn_like(primary_action_mask) * primary_action_mask

        dt = 1.0 / self.num_inference_timesteps
        vel_strength = primary_action_mask.clone()

        if "action" in action_input:
            # If action in input when doing get action, it means we want to use RTC.
            # action_horizon is the action horizon of the input action.
            # rtc_overlap_steps is the number of steps to overlap with the previous action chunks.
            # rtc_frozen_steps is the number of steps to freeze the action, which is the latency of the policy inference.
            # rtc_ramp_rate is the rate of the ramp of denoising the actions.
            assert options is not None, "options is not None"
            assert "action_horizon" in options, "action_horizon is not in options"
            assert "rtc_overlap_steps" in options, "rtc_overlap_steps is not in options"
            assert "rtc_frozen_steps" in options, "rtc_frozen_steps is not in options"
            assert "rtc_ramp_rate" in options, "rtc_ramp_rate is not in options"

            action_horizon_before_padding = options["action_horizon"]

            # Use previous action instead of pure noise to do inpainting
            actions[:, : options["rtc_overlap_steps"], :] = action_input["action"][
                :,
                action_horizon_before_padding
                - options["rtc_overlap_steps"] : action_horizon_before_padding,
                :,
            ]
            vel_strength[:, : options["rtc_frozen_steps"], :] = 0.0
            # NOTE: use an exponential ramp strength to set the remaining unfrozen rtc_steps
            intermediate_steps = options["rtc_overlap_steps"] - options["rtc_frozen_steps"]
            # Create exponential ramp from 0 to 1 over intermediate steps
            t = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
            ramp = 1 - torch.exp(-options["rtc_ramp_rate"] * t)
            ramp = ramp / ramp[-1].clamp_min(1e-8)  # normalize to [0,1]
            ramp = ramp[
                1:-1
            ]  # we will only take the middle part of the ramp, ignore the 0.0 and 1.0
            # Apply ramp to the intermediate steps [batch, intermediate_steps, action_dim]
            vel_strength[
                :,
                options["rtc_frozen_steps"] : options["rtc_overlap_steps"],
                :,
            ] = ramp[None, :, None].to(device)
            actions = actions * primary_action_mask
            vel_strength = vel_strength * primary_action_mask

        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward.
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :] * primary_action_mask

            # Update actions using euler integration.
            actions = (actions + dt * pred_velocity * vel_strength) * primary_action_mask

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
            options=options,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def _is_qwen3_vl_model_name(model_name: str) -> bool:
    model_name = str(model_name)
    normalized_name = model_name.lower()
    if any(token in normalized_name for token in ("cosmos-reason2", "qwen3-vl", "qwen3_vl")):
        return True

    config_path = Path(model_name) / "config.json"
    if not config_path.is_file():
        return False

    try:
        model_type = str(json.loads(config_path.read_text()).get("model_type", "")).lower()
    except Exception:
        return False
    return model_type == "qwen3_vl"


def get_backbone_cls(config: Gr00tN1d7Config):
    if _is_qwen3_vl_model_name(config.model_name):
        # We import here as Qwen3Backbone depends on newer transformers versions than the rest of the code.
        from gr00t.model.modules.qwen3_backbone import Qwen3Backbone

        return Qwen3Backbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d7(PreTrainedModel):
    """Gr00tN1d7: VLA model with Cosmos-Reason2-2B (Qwen3-VL) backbone."""

    config_class = Gr00tN1d7Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d7Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d7 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d7ActionHead(config)
        if config.task_status_head_only and not config.task_status_head_enabled:
            raise ValueError("task_status_head_only requires task_status_head_enabled=True")
        if config.task_status_completion_window_frames <= 0:
            raise ValueError("task_status_completion_window_frames must be positive")
        if (
            not math.isfinite(config.task_status_completion_pos_weight)
            or config.task_status_completion_pos_weight <= 0
        ):
            raise ValueError("task_status_completion_pos_weight must be positive")
        for name in (
            "task_status_loss_weight",
            "task_status_completion_loss_weight",
            "task_status_progress_loss_weight",
        ):
            value = getattr(config, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        if (
            config.task_status_completion_loss_weight == 0
            and config.task_status_progress_loss_weight == 0
        ):
            raise ValueError("At least one task-status objective weight must be positive")
        for name in (
            "task_status_completion_threshold",
            "task_status_progress_threshold",
        ):
            value = getattr(config, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1], got {value}")
        if config.task_status_consecutive_steps <= 0:
            raise ValueError("task_status_consecutive_steps must be positive")
        self.task_status_head = None
        if config.task_status_head_enabled:
            self.task_status_head = Gr00tTaskStatusHead(
                embedding_dim=config.backbone_embedding_dim,
                hidden_dim=config.task_status_hidden_dim,
                num_heads=config.task_status_num_heads,
                dropout=config.task_status_dropout,
            )
        if config.task_status_head_only:
            # Enforce the mode at module level as well as through individual tune
            # flags, so tied or newly-added policy parameters cannot stay trainable.
            self.backbone.requires_grad_(False)
            self.action_head.requires_grad_(False)
            self.task_status_head.requires_grad_(True)
        from .processing_gr00t_n1d7 import Gr00tN1d7DataCollator

        self.collator = Gr00tN1d7DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def _get_task_status_outputs(
        self,
        backbone_outputs: BatchFeature,
        action_inputs: BatchFeature,
    ) -> BatchFeature:
        if self.task_status_head is None:
            return BatchFeature()

        logits = self.task_status_head(
            backbone_outputs.backbone_features,
            backbone_outputs.get("backbone_attention_mask"),
        )
        completion_logit = logits[:, 0]
        progress_logit = logits[:, 1]
        outputs = BatchFeature(
            data={
                "completion_logit": completion_logit,
                "progress_logit": progress_logit,
                "completion": torch.sigmoid(completion_logit),
                "progress": torch.sigmoid(progress_logit),
            }
        )

        frame_index = action_inputs.get("frame_index")
        episode_length = action_inputs.get("episode_length")
        if frame_index is None or episode_length is None:
            if self.training:
                raise ValueError(
                    "Training with task_status_head_enabled=True requires frame_index "
                    "and episode_length inputs"
                )
            return outputs

        frame_index = frame_index.reshape(-1).to(device=logits.device, dtype=torch.long)
        episode_length = episode_length.reshape(-1).to(device=logits.device, dtype=torch.long)
        if (
            frame_index.shape != completion_logit.shape
            or episode_length.shape != completion_logit.shape
        ):
            raise ValueError(
                "frame_index and episode_length must contain one value per sample, got "
                f"{tuple(frame_index.shape)}, {tuple(episode_length.shape)}, and "
                f"batch {tuple(completion_logit.shape)}"
            )
        if torch.any(episode_length <= 0):
            raise ValueError("episode_length must be positive")
        if torch.any(frame_index < 0) or torch.any(frame_index >= episode_length):
            raise ValueError("frame_index must be in [0, episode_length)")

        terminal_window = torch.minimum(
            torch.full_like(
                episode_length,
                self.config.task_status_completion_window_frames,
            ),
            episode_length.sub(1).clamp_min(0),
        )
        terminal_start = episode_length - terminal_window
        completion_target = (frame_index >= terminal_start).to(dtype=torch.float32)
        # The terminal window is a completed hold state: progress reaches one at
        # its first frame and remains there through the end of the episode.
        progress_target = (
            frame_index.to(dtype=torch.float32)
            / terminal_start.clamp_min(1).to(dtype=torch.float32)
        ).clamp(max=1.0)

        pos_weight = torch.tensor(
            self.config.task_status_completion_pos_weight,
            device=completion_logit.device,
            dtype=torch.float32,
        )
        completion_loss = F.binary_cross_entropy_with_logits(
            completion_logit.float(),
            completion_target,
            pos_weight=pos_weight,
        )
        progress_loss = F.smooth_l1_loss(outputs.progress.float(), progress_target)
        status_loss = (
            self.config.task_status_completion_loss_weight * completion_loss
            + self.config.task_status_progress_loss_weight * progress_loss
        )
        outputs.update(
            {
                "completion_target": completion_target,
                "progress_target": progress_target,
                "completion_loss": completion_loss,
                "progress_loss": progress_loss,
                "task_status_loss": status_loss,
            }
        )
        return outputs

    def forward(self, inputs: dict) -> dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            Dictionary containing loss and other outputs. Hugging Face Trainer
            requires a dictionary (or tuple/ModelOutput) so it can retrieve the
            loss without treating the result as an integer-indexable sequence.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        status_outputs = self._get_task_status_outputs(backbone_outputs, action_inputs)

        if self.config.task_status_head_only:
            if "task_status_loss" not in status_outputs:
                raise ValueError(
                    "Task-status-only training requires frame_index and episode_length inputs"
                )
            status_outputs["loss"] = status_outputs.task_status_loss
            return dict(status_outputs)

        action_outputs = self.action_head(backbone_outputs, action_inputs)

        if "task_status_loss" in status_outputs:
            action_outputs["action_policy_loss"] = action_outputs.loss
            action_outputs["loss"] = (
                action_outputs.loss
                + self.config.task_status_loss_weight * status_outputs.task_status_loss
            )
        action_outputs.update(status_outputs)

        return action_outputs

    def get_action(self, inputs: dict, options: dict[str, Any] | None = None) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        status_outputs = self._get_task_status_outputs(backbone_outputs, action_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs, options)
        action_outputs.update(status_outputs)

        return action_outputs

    @torch.no_grad()
    def get_task_status(self, inputs: dict) -> BatchFeature:
        """Predict completion and progress without running the action diffusion head."""
        if self.task_status_head is None:
            raise RuntimeError("This checkpoint does not contain a task-status head")

        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        return self._get_task_status_outputs(backbone_outputs, action_inputs)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)
