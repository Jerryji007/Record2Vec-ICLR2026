import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import math

from torch.optim.lr_scheduler import LambdaLR

import torch.nn.functional as F

from utils.shape_alignment import _pad_truncate_to, _overlap_mask_like, _broadcast_base_mask

def _pad_truncate_to(x: torch.Tensor, target_shape: torch.Size, pad_value: float = 0.0) -> torch.Tensor:
    """
    Pad (right) or truncate tensor x to match target_shape (including batch dim).
    Works for 2D/3D+ tensors; pads only on the *end* of each dimension.
    """
    assert x.dim() == len(target_shape), f"x.dim={x.dim()} vs target_shape={target_shape}"
    out = x

    slicers = []
    for d, tsize in enumerate(target_shape):
        csize = out.size(d)
        slicers.append(slice(0, min(csize, tsize)))
    out = out[tuple(slicers)]

    pads = []
    for d in reversed(range(out.dim())):
        csize = out.size(d)
        tsize = target_shape[d]
        add = max(tsize - csize, 0)
        pads.extend([0, add])  # pad only at the end
    if any(pads):
        out = F.pad(out, pads, mode="constant", value=pad_value)

    slicers = []
    for d, tsize in enumerate(target_shape):
        slicers.append(slice(0, tsize))
    return out[tuple(slicers)]


def _overlap_mask_like(targets: torch.Tensor, orig_shape: torch.Size) -> torch.Tensor:
    """
    Build a mask (same shape as targets) with 1's in the region that overlaps with the original
    model output (before padding), and 0's elsewhere (the padded/unavailable part).
    Only considers dims after the batch dim.
    """
    device = targets.device
    dtype  = targets.dtype
    mask = torch.zeros_like(targets, dtype=dtype, device=device)

    if targets.dim() == 2:
        B, Tt = targets.shape
        To = orig_shape[1]
        T = min(Tt, To)
        mask[:, :T] = 1
    elif targets.dim() == 3:
        B, Tt, Dt = targets.shape
        To = orig_shape[1]
        Do = orig_shape[2]
        T = min(Tt, To)
        D = min(Dt, Do)
        mask[:, :T, :D] = 1
    else:
        slicers = [slice(None)]
        for d in range(1, targets.dim()):
            slicers.append(slice(0, min(targets.size(d), orig_shape[d])))
        mask[tuple(slicers)] = 1

    return mask

def masked_mse(pred, target, mask, eps=1e-8):
    # mask: bool or {0,1}, same shape as target
    m = mask.to(dtype=pred.dtype)
    se = F.mse_loss(pred, target, reduction='none')  # [B, D, T]
    num = (se * m).sum()
    den = m.sum().clamp_min(eps)
    return num / den


class TemporalDecoder(nn.Module):
    """
    Map a single embedding vector [B, E] to a time-varying sequence [B, S, F]
    using additive conditioning with learnable positional embeddings.
    """
    def __init__(self, E: int, F: int, S: int, H: int = 512, learned_pos: bool = True, p_drop: float = 0.1):
        super().__init__()
        self.S = S
        self.E = E
        self.F = F
        self.H = H

        self.e_proj = nn.Linear(E, H)
        self.norm  = nn.LayerNorm(H)
        self.drop  = nn.Dropout(p_drop)
        self.out   = nn.Linear(H, F)

        if learned_pos:
            self.pos = nn.Parameter(torch.zeros(S, H))
            nn.init.normal_(self.pos, std=0.02)
        else:
            self.register_buffer("pos", self._build_sinusoid(S, H), persistent=False)

        self.act = nn.ReLU(inplace=True)

    @staticmethod
    def _build_sinusoid(S, H):
        pe = torch.zeros(S, H)
        position = torch.arange(0, S, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, H, 2, dtype=torch.float32) * (-math.log(10000.0) / H))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, emb_flat: torch.Tensor) -> torch.Tensor:
        """
        emb_flat: [B, E]
        returns:  [B, S, F]
        """
        B = emb_flat.size(0)
        h = self.e_proj(emb_flat)                       # [B, H]
        h = h.unsqueeze(1) + self.pos.unsqueeze(0)      # [B, S, H]
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        dec = self.out(h)                               # [B, S, F]
        return dec


class Trainer:
    def __init__(self, args, model, task_config):
        """
        Args:
            args: Command-line arguments.
            model: The initialized model.
            task_config (dict): Dictionary containing configuration parameters.
                Required keys:
                    - dataset
                    - task
                    - learning_rate
                    - batch_size
                    - patience
                    - train_mask
                    - model_type
                    - seed
                    - representation
                    - embed_dim
        """
        self.args = args
        self.model = model
        self.task_config = task_config
        
        self.llm_type = task_config.get("embedding_model", None)
        self.summarization_model = task_config.get("summarization_model", "")

        if hasattr(args, "use_temporal"):
            self.use_temporal = args.use_temporal
        else:
            self.use_temporal = False
        
        if self.llm_type:
            if self.use_temporal:
                E = task_config["embed_dim"]
                H = task_config.get("decoder_hidden", 2048)
                F_dec = task_config.get("decoder_features", 75)
                self.F = F_dec
                self.S = 48

                self.decoder = TemporalDecoder(E=E, F=F_dec, S=self.S, H=min(H, 512), learned_pos=True, p_drop=0.1)
            else:
                E = task_config["embed_dim"] 
                H = task_config.get("decoder_hidden", 2048) 
                # option1: summary all features -> 100 # time * features 
                F_dec = task_config.get("decoder_features", 75) 
                self.F = F_dec 
                self.S = 48 # result: 75 * 1 vector -> replicate 
                self.decoder = nn.Sequential(
                    nn.Linear(E, H), 
                    nn.ReLU(inplace=True),
                    nn.Linear(H, F_dec)
                )

        self.task = task_config["task"]
        self.dataset = task_config["dataset"]

        # Set up loss function based on the task.
        if self.task == "forecast":
            self.loss_fn = masked_mse
        elif self.task in ["mort"]:
            self.loss_fn = None
        elif self.task in ["gender"]:
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif self.task in ["los", "age"]:
            self.loss_fn = nn.MSELoss()
        elif self.task in ["lab", "feature"]:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        self.learning_rate = task_config["learning_rate"]
        self.batch_size = task_config["batch_size"]
        self.finetune_size = task_config.get("finetune_size", self.batch_size)
        self.patience = task_config["patience"]
        self.seed = task_config["seed"]
        self.model_type = task_config["model_type"]
        self.representation = task_config["representation"]
        self.train_mask = task_config.get("train_mask", "all")

        # Build experiment string.
        if self.representation in ["all", "sbf", "sbt", "all_sum"]:
            self.representation = f"{self.representation}_{self.llm_type}"
        lr_str = f"{self.learning_rate:.0e}"
        if self.use_temporal:
            self.experiment_string = f"{self.model_type}_{self.representation}_{self.train_mask}_{lr_str}_{self.batch_size}_seed_{self.seed}_temp_{self.use_temporal}"
        else:
            self.experiment_string = f"{self.model_type}_{self.representation}_{self.train_mask}_{lr_str}_{self.batch_size}_seed_{self.seed}"

        # Create a directory for dataset/task/sum_model.
        if args.embedding_model != "Qwen3-Embedding-8B+TFM":
            self.save_dir = os.path.join("", self.dataset, self.task, self.summarization_model)
        else:
            self.save_dir = os.path.join("", self.dataset, self.task, self.summarization_model, args.embedding_model)

        os.makedirs(self.save_dir, exist_ok=True)

        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

        # Set the device based on the provided GPU argument.
        # device = (
        #     torch.device("cuda:{}".format(self.args.gpu))
        #     if torch.cuda.is_available()
        #     else torch.device("cpu")
        # )
        # self.device = device
        # print(f"Using device: {self.device}")
        self.device = self._pick_device()
        self.model.to(self.device)
        if self.llm_type:
            self.decoder.to(self.device)

        self.model.to(self.device)

        if self.llm_type:
            self.decoder.to(self.device)

            self.optimizer = optim.Adam(
                list(self.model.parameters()) + list(self.decoder.parameters()),
                lr=task_config["learning_rate"]
            )
        else:
            self.optimizer = optim.Adam(
                list(self.model.parameters()),
                lr=task_config["learning_rate"]
            )

    def _pick_device(self):
        if not torch.cuda.is_available():
            print("[INFO] CUDA not available → using CPU")
            return torch.device("cpu")

        n = torch.cuda.device_count()
        idx = getattr(self.args, "gpu", 0)
        try:
            idx = int(idx)
        except Exception:
            idx = 0

        if idx < 0 or idx >= n:
            print(f"[WARN] Requested gpu index {idx} but only {n} device(s) visible "
                f"({os.environ.get('CUDA_VISIBLE_DEVICES','unset')}); falling back to 0")
            idx = 0

        print(f"[INFO] Using CUDA device cuda:{idx} of {n} visible")
        return torch.device(f"cuda:{idx}")

    def _compute_pos_weight(self, loader):
        N1 = 0.0
        N0 = 0.0
        with torch.no_grad():
            for batch in loader:
                t = self._get_targets(batch)
                t = t.to(self.device).float().view(-1)
                N1 += float(t.sum().item())
                N0 += float(t.numel() - t.sum().item())
        total = N0 + N1 if (N0 + N1) > 0 else 1.0
        p = N1 / total
        pos_w = N0 / max(N1, 1.0)  # weight positives by inverse frequency
        pw = torch.tensor([pos_w], device=self.device, dtype=torch.float32)
        return float(p), pw
    
    def _set_final_bias(self, base_rate):
        """Initialize the last Linear(out_features==1) bias to logit(base_rate)."""
        q = min(max(base_rate, 1e-6), 1 - 1e-6)
        bias_val = math.log(q / (1.0 - q))
        for m in self.model.modules():
            if isinstance(m, nn.Linear) and m.out_features == 1 and m.bias is not None:
                with torch.no_grad():
                    m.bias.data.fill_(bias_val)
                print(f"[INFO] Initialized final bias to logit(base_rate)={bias_val:.4f} (p={q:.4f})")
                break

    @staticmethod
    def _looks_like_probs(t: torch.Tensor) -> bool:
        with torch.no_grad():
            return bool(t.min() >= 0.0 and t.max() <= 1.0)

    def _get_targets(self, batch):
        """
        Select the correct target from the batch based on the task.
        Batch indices:
            index 0: inputs
            index 1: forecast target (if forecast)
            index 2: forecast mask (if forecast)
            index 3: mort label
            index 4: los label
            index 5: sepsis label
            index 6: aki label
            index 7: kf label
        """
        if self.task == "forecast":
            return batch[1].float().to(self.device)
        elif self.task == "mort":
            targets = batch[3].float().to(self.device)
        elif self.task == "los":
            targets = batch[4].to(self.device)
            if targets.dim() == 1:
                targets = targets.unsqueeze(-1)
        elif self.task == "lab":
            targets = batch[5].float().to(self.device)
        elif self.task == "feature":
            targets = batch[6].float().to(self.device)
        elif self.task == "gender":
            targets = batch[7].float().to(self.device)
        elif self.task == "age":
            targets = batch[8].float().to(self.device)
        else:
            raise ValueError("Unsupported task in target selection")

        # For classification tasks, if the target is 1D, unsqueeze to match the output shape [batch_size, 1]
        if self.task in ["mort", "lab", "feature", "gender"] and targets.dim() == 1:
            targets = targets.unsqueeze(1)
        return targets

    def parse_model_filename(self, filename):
        """
        Parses a filename of the form:
        For normal representations:
            best_{model}_{representation}_{train_mask}_{lr}_{batch_size}_seed_{seed}.pt
        For LLM-based representations:
            best_{model}_{representation}_{llm_type}_{train_mask}_{lr}_{batch_size}_seed_{seed}.pt
        where the llm_type string may itself contain underscores.

        Returns a dictionary with the following keys:
        model, representation, llm_type, train_mask, lr, bs, seed
        """
        # Remove "best_" prefix and ".pt" suffix.
        base = filename.replace("best_", "").replace(".pt", "")

        # Split off the seed part (assumes '_seed_' is present)
        if "_seed_" not in base:
            print(
                f"Filename {filename} does not contain '_seed_' and cannot be parsed."
            )
            return None
        base_without_seed, seed_str = base.rsplit("_seed_", 1)
        seed = seed_str

        # Now, the beginning of base_without_seed has model and representation.
        # Do a left-split into three parts so that the first two fields are isolated.
        parts = base_without_seed.split("_", 2)
        if len(parts) < 3:
            print(f"Filename {filename} does not match expected pattern.")
            return None
        model = parts[0]
        representation = parts[1]
        remainder = parts[2]  # This contains either:
        #   (a) {train_mask}_{lr}_{bs} for normal representations, or
        #   (b) {llm_type}_{train_mask}_{lr}_{bs} for LLM-based representations.

        # Right-split the remainder into the last three tokens.
        tail_tokens = remainder.rsplit("_", 3)
        if len(tail_tokens) == 4:
            llm_type = tail_tokens[0]
            train_mask = tail_tokens[1]
            lr = tail_tokens[2]
            bs = tail_tokens[3]
        elif len(tail_tokens) == 3:
            llm_type = None
            train_mask, lr, bs = tail_tokens
        else:
            print(f"Filename {filename} does not match expected pattern.")
            return None
        self.args.seed = int(seed)
        return {
            "model": model,
            "representation": representation,
            "llm_type": llm_type,
            "train_mask": train_mask,
            "lr": lr,
            "bs": bs,
            "seed": seed,
        }


    def finetune(self, model, decoder, train_loader, val_loader):
        if self.loss_fn is None and self.task in ["mort"]:
            base_rate, pos_w = self._compute_pos_weight(train_loader)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
            print(f"[INFO] {self.task}: base_rate={base_rate:.4f}, pos_weight={pos_w.item():.3f}")
            self._set_final_bias(base_rate)

        self.model = model
        self.decoder = decoder

        if self.task == "forecast":
            if self.llm_type and self.llm_type != "TFM":
                num_epochs = 1000
            else:
                num_epochs = 1000
        else:
            num_epochs = 100

        REF_FINETUNE_BS   = 16
        BASE_LR           = 1e-6
        BASE_WD           = 0.01
        MIN_LR_RATIO      = 0.10
        WARMUP_FRAC       = 0.03

        eff_bs = self.batch_size
        scaled_lr = BASE_LR * (eff_bs / REF_FINETUNE_BS)
        wd = float(BASE_WD)

        num_batches = len(train_loader)
        total_steps = max(1, num_epochs * num_batches)

        warmup_steps = max(10, int(WARMUP_FRAC * total_steps))

        for p in self.model.parameters():
            p.requires_grad = True

        if self.llm_type:
            for p in self.decoder.parameters():
                p.requires_grad = True

        if self.llm_type:
            self.optimizer = optim.AdamW(
                params=list(self.model.parameters()) + list(self.decoder.parameters()),
                # params=list(self.decoder.parameters()),
                lr=scaled_lr,
                weight_decay=0.01,
                betas=(0.9, 0.999)
            )
        else:
            self.optimizer = optim.AdamW(
                params=list(self.model.parameters()),
                lr=1e-6,
                weight_decay=0.01,
                betas=(0.9, 0.999)
            )

        steps_per_epoch = max(1, len(train_loader))
        total_steps = max(1, num_epochs * steps_per_epoch)
        warmup_steps = min(3, total_steps - 1) if total_steps > 1 else 0

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / float(max(1, warmup_steps))
            denom = max(1, total_steps - warmup_steps)
            t = (step - warmup_steps) / denom
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * 0.5 * (1.0 + math.cos(math.pi * t))

        scheduler = LambdaLR(self.optimizer, lr_lambda)

        best_model_weights = self.model.state_dict()
        if self.llm_type:
            best_decoder_weights = self.decoder.state_dict()
        else:
            best_decoder_weights = None

        best_loss = float("inf")
        global_step = 0

        val_loss = 0.0
        # init a validation loss
        with torch.no_grad():
            for batch in val_loader:
                emb = batch[0].to(self.device).float()

                if self.llm_type: 
                    if not self.use_temporal:
                        emb_flat = emb.view(emb.size(0), -1) 
                        dec_vec = self.decoder(emb_flat) 
                        dec = dec_vec.unsqueeze(1).expand(-1, self.S, -1) 
                    else:
                        emb_flat = emb.view(emb.size(0), -1)
                        dec = self.decoder(emb_flat)
                else:
                    dec = emb.float()

                outputs = self.model(dec)
                targets = self._get_targets(batch)

                if self.task == "forecast":
                    orig_out_shape = tuple(outputs.shape)
                    outputs_aligned = _pad_truncate_to(outputs, targets.shape, pad_value=0.0)

                    base_mask = batch[2].to(self.device).to(dtype=targets.dtype).contiguous()
                    base_mask = _broadcast_base_mask(base_mask, targets)

                    overlap_mask = _overlap_mask_like(targets, orig_out_shape).to(targets.dtype)

                    final_mask = (base_mask > 0.5).to(targets.dtype) * overlap_mask

                if self.task != "forecast":
                    loss = self.loss_fn(outputs, targets)
                else:
                    loss = self.loss_fn(outputs_aligned, targets, final_mask)

                val_loss += loss.item() * emb.size(0)

            epoch_val_loss = val_loss / len(val_loader.dataset)
            self.best_val_loss = epoch_val_loss

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            if self.llm_type:
                self.decoder.train()

            running_loss = 0.0
            n_seen = 0

            for batch in train_loader:
                self.optimizer.zero_grad()

                emb = batch[0].to(self.device).float()

                # keep embeddings frozen; decoder is frozen too
                if self.llm_type:
                    if not self.use_temporal:
                        emb_flat = emb.view(emb.size(0), -1)
                        dec_vec = self.decoder(emb_flat)
                        dec = dec_vec.unsqueeze(1).expand(-1, self.S, -1)
                    else:
                        emb_flat = emb.view(emb.size(0), -1)
                        dec = self.decoder(emb_flat)
                else:
                    dec = emb.float()

                outputs = self.model(dec)
                targets = self._get_targets(batch)

                if self.task == "forecast":
                    orig_out_shape = tuple(outputs.shape)
                    outputs_aligned = _pad_truncate_to(outputs, targets.shape, pad_value=0.0)

                    base_mask = batch[2].to(self.device).to(dtype=targets.dtype).contiguous()
                    base_mask = _broadcast_base_mask(base_mask, targets)

                    overlap_mask = _overlap_mask_like(targets, orig_out_shape).to(targets.dtype)

                    final_mask = (base_mask > 0.5).to(targets.dtype) * overlap_mask

                if self.task != "forecast":
                    loss = self.loss_fn(outputs, targets)
                else:
                    loss = self.loss_fn(outputs_aligned, targets, final_mask)

                loss.backward()
                self.optimizer.step()
                scheduler.step()

                bs = emb.size(0)
                running_loss += loss.item() * bs
                n_seen += bs
                global_step += 1
        
                epoch_train_loss = running_loss / max(1, n_seen)

            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    emb = batch[0].to(self.device).float()

                    if self.llm_type: 
                        if not self.use_temporal:
                            emb_flat = emb.view(emb.size(0), -1) 
                            dec_vec = self.decoder(emb_flat) 
                            dec = dec_vec.unsqueeze(1).expand(-1, self.S, -1) 
                        else:
                            emb_flat = emb.view(emb.size(0), -1)
                            dec = self.decoder(emb_flat)
                    else:
                        dec = emb.float()

                    outputs = self.model(dec)
                    targets = self._get_targets(batch)
                    
                    if self.task == "forecast":
                        orig_out_shape = tuple(outputs.shape)
                        outputs_aligned = _pad_truncate_to(outputs, targets.shape, pad_value=0.0)

                        base_mask = batch[2].to(self.device).to(dtype=targets.dtype).contiguous()
                        base_mask = _broadcast_base_mask(base_mask, targets)

                        overlap_mask = _overlap_mask_like(targets, orig_out_shape).to(targets.dtype)

                        final_mask = (base_mask > 0.5).to(targets.dtype) * overlap_mask
                    
                    if self.task != "forecast":
                        loss = self.loss_fn(outputs, targets)
                    else:
                        loss = self.loss_fn(outputs_aligned, targets, final_mask)

                    val_loss += loss.item() * emb.size(0)

                epoch_val_loss = val_loss / len(val_loader.dataset)

                if epoch % 100 == 0:
                    print(f"Epoch {epoch}/{num_epochs} Validation Loss: {epoch_val_loss:.4f}")
            
            if epoch_val_loss < self.best_val_loss:
                best_model_weights = self.model.state_dict()
                if self.llm_type:
                    best_decoder_weights = self.decoder.state_dict()
                else:
                    best_decoder_weights = None
                self.best_val_loss = epoch_val_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                # print(f"No improvement for {self.epochs_no_improve} epoch(s).")
                if self.epochs_no_improve >= 15:
                    # print("Early stopping triggered.")
                    break

        return best_decoder_weights, best_model_weights


    def train(self, train_loader, val_loader):
        best_model_weights = None
        num_epochs = self.args.num_epochs

        if self.loss_fn is None and self.task in ["mort"]:
            base_rate, pos_w = self._compute_pos_weight(train_loader)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
            print(f"[INFO] {self.task}: base_rate={base_rate:.4f}, pos_weight={pos_w.item():.3f}")
            self._set_final_bias(base_rate)

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            running_loss = 0.0

            for batch in train_loader:
                self.optimizer.zero_grad()

                emb = batch[0].to(self.device).float()
                
                if self.llm_type:
                    if not self.use_temporal:
                        emb_flat = emb.view(emb.size(0), -1) 
                        dec_vec = self.decoder(emb_flat) 
                        dec = dec_vec.unsqueeze(1).expand(-1, self.S, -1) 
                    else:
                        emb_flat = emb.view(emb.size(0), -1)
                        dec = self.decoder(emb_flat)
                else:
                    dec = emb.float()
                    
                targets = self._get_targets(batch)
                outputs = self.model(dec)

                if self.task != "forecast":
                    loss = self.loss_fn(outputs, targets)
                else:
                    mask = batch[2].to(self.device).to(dtype=targets.dtype).contiguous()
                    loss = self.loss_fn(outputs, targets, mask)

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * emb.size(0)

            epoch_train_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch {epoch}/{num_epochs} Training Loss: {epoch_train_loss:.4f}")

            # Validation loop.
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    emb = batch[0].to(self.device).float()

                    if self.llm_type: 
                        if not self.use_temporal:
                            emb_flat = emb.view(emb.size(0), -1) 
                            dec_vec = self.decoder(emb_flat) 
                            dec = dec_vec.unsqueeze(1).expand(-1, self.S, -1) 
                        else:
                            emb_flat = emb.view(emb.size(0), -1)
                            dec = self.decoder(emb_flat)
                    else:
                        dec = emb.float()

                    outputs = self.model(dec)
                    targets = self._get_targets(batch)

                    if self.task != "forecast":
                        loss = self.loss_fn(outputs, targets)
                    else:
                        mask = batch[2].to(self.device).to(dtype=targets.dtype).contiguous()
                        loss = self.loss_fn(outputs, targets, mask)

                    val_loss += loss.item() * emb.size(0)

            epoch_val_loss = val_loss / len(val_loader.dataset)
            print(f"Epoch {epoch}/{num_epochs} Validation Loss: {epoch_val_loss:.4f}")

            # Save model if validation loss improved.
            if epoch_val_loss < self.best_val_loss:
                self.best_val_loss = epoch_val_loss
                best_model_weights = self.model.state_dict()
                if self.llm_type:
                    decoder_sd = self.decoder.state_dict()
                else:
                    decoder_sd = None
                self.epochs_no_improve = 0
                save_path = os.path.join(
                    self.save_dir, f"best_{self.experiment_string}.pt"
                )

                ckpt = {
                    "model_state": best_model_weights,
                    "decoder_state": decoder_sd
                }
                torch.save(ckpt, save_path)
                print(f"Validation loss improved; model saved to {save_path}")
            else:
                self.epochs_no_improve += 1
                print(f"No improvement for {self.epochs_no_improve} epoch(s).")
                if self.epochs_no_improve >= self.patience:
                    print("Early stopping triggered.")
                    break

        print("Training finished.")


    def finetune_simple(self, model, decoder, train_loader, val_loader):
        """
        Simple finetuning + early stopping:
        - constant LR = 1e-5
        - no scheduler, no weight decay
        - up to 100 epochs
        - early stopping on validation loss with patience=8
        Returns: (best_decoder_sd, best_model_sd)
        """
        from copy import deepcopy
        import math
        import torch

        # Loss setup (unchanged)
        if self.loss_fn is None and self.task in ["mort"]:
            base_rate, pos_w = self._compute_pos_weight(train_loader)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
            print(f"[INFO] {self.task}: base_rate={base_rate:.4f}, pos_weight={pos_w.item():.3f}")
            self._set_final_bias(base_rate)

        self.model = model
        self.decoder = decoder

        # Trainable params (same logic)
        for p in self.model.parameters():
            p.requires_grad = True
        if self.llm_type:
            for p in self.decoder.parameters():
                p.requires_grad = True

        # Optimizer: constant LR, no decay
        self.optimizer = optim.SGD(
            list(self.model.parameters()) + list(self.decoder.parameters()),
            lr=1e-5, weight_decay=0.0
        )

        num_epochs = 200
        global_step = 0

        # Early stopping state
        patience = 8
        best_val_loss = math.inf
        best_model_sd = deepcopy(self.model.state_dict())
        best_decoder_sd = deepcopy(self.decoder.state_dict()) if self.llm_type else None
        epochs_no_improve = 0

        def _compute_batch_loss(outputs, targets, batch):
            if self.task == "forecast":
                orig_out_shape = tuple(outputs.shape)
                outputs_aligned = _pad_truncate_to(outputs, targets.shape, pad_value=0.0)

                base_mask = batch[2].to(self.device).to(dtype=targets.dtype).contiguous()
                base_mask = _broadcast_base_mask(base_mask, targets)

                overlap_mask = _overlap_mask_like(targets, orig_out_shape).to(targets.dtype)
                final_mask = (base_mask > 0.5).to(targets.dtype) * overlap_mask
                return self.loss_fn(outputs_aligned, targets, final_mask)
            else:
                return self.loss_fn(outputs, targets)

        # Training loop
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            if self.llm_type:
                self.decoder.train()

            running_loss = 0.0
            n_seen = 0

            for batch in train_loader:
                self.optimizer.zero_grad()

                emb = batch[0].to(self.device).float()

                if self.llm_type:
                    if not self.use_temporal:
                        emb_flat = emb.view(emb.size(0), -1)
                        dec_vec = self.decoder(emb_flat)
                        dec = dec_vec.unsqueeze(1).expand(-1, self.S, -1)
                    else:
                        emb_flat = emb.view(emb.size(0), -1)
                        dec = self.decoder(emb_flat)
                else:
                    dec = emb.float()

                outputs = self.model(dec)
                targets = self._get_targets(batch)
                loss = _compute_batch_loss(outputs, targets, batch)

                loss.backward()
                self.optimizer.step()

                bs = emb.size(0)
                running_loss += loss.item() * bs
                n_seen += bs
                global_step += 1

            epoch_train_loss = running_loss / max(1, n_seen)

            # Logging
            if epoch % 20 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] Train {epoch_train_loss:.6f}")

        # Return best weights seen on validation
        return best_decoder_sd, best_model_sd

