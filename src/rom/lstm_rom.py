"""
LSTM-based Reduced Order Model (ROM) for latent dynamics prediction.

This module implements a PyTorch LSTM model that operates in the POD latent space.
The model takes a sequence of past latent states and predicts the next latent state.

Architecture:
    Input: [batch_size, lag, d] - sequence of latent states
    LSTM: Processes temporal sequences
    Output: [batch_size, d] - predicted next latent state

Usage:
    from src.rom.lstm_rom import LatentLSTMROM
    
    model = LatentLSTMROM(d=25, hidden_units=16, num_layers=1)
    x_seq = torch.randn(batch_size, lag, d)
    y_pred = model(x_seq)  # [batch_size, d]
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LatentLSTMROM(nn.Module):
    """
    LSTM-based Reduced Order Model for latent dynamics.
    
    This model learns to predict the next latent state given a sequence
    of past latent states in the POD-reduced space.
    
    Parameters
    ----------
    d : int
        Latent dimension (number of POD modes).
    hidden_units : int, optional
        Number of LSTM hidden units (Nh). Default: 16.
    num_layers : int, optional
        Number of stacked LSTM layers. Default: 1.
    
    Attributes
    ----------
    d : int
        Latent dimension.
    hidden_units : int
        Number of LSTM hidden units.
    lstm : nn.LSTM
        LSTM layer(s) for sequence processing.
    out : nn.Linear
        Linear layer mapping LSTM output to latent prediction.
    
    Examples
    --------
    >>> model = LatentLSTMROM(d=25, hidden_units=32, num_layers=2)
    >>> x = torch.randn(16, 10, 25)  # [batch=16, lag=10, d=25]
    >>> y_pred = model(x)             # [16, 25]
    >>> y_pred.shape
    torch.Size([16, 25])
    """
    
    def __init__(self, d, hidden_units=16, num_layers=1, dropout=0.0, 
                 residual=False, use_layer_norm=True):
        """
        Initialize the LatentLSTMROM model.
        
        Parameters
        ----------
        d : int
            Latent dimension (number of POD modes).
        hidden_units : int, optional
            Number of LSTM hidden units (Nh). Default: 16.
        num_layers : int, optional
            Number of stacked LSTM layers. Default: 1.
        dropout : float, optional
            Dropout probability between LSTM layers. Default: 0.0.
            Only applied if num_layers > 1.
        residual : bool, optional
            If True, model predicts Δy = y(t+1) - y(t) and adds it to the last
            input state (residual/delta formulation). This lets the network focus
            on learning the dynamics rather than the identity. Default: False.
        use_layer_norm : bool, optional
            If True, apply LayerNorm to the LSTM hidden state before the output
            linear layer. Generally stabilizing, but if inputs are already
            z-scored per mode, this can sometimes hurt slightly. Default: True.
        """
        super().__init__()
        self.d = d
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.residual = residual
        self.use_layer_norm = use_layer_norm
        
        # LSTM block:
        #   input_size  = d  (latent dimension)
        #   hidden_size = hidden_units
        #   num_layers  = num_layers
        #   dropout     = dropout (between layers, only if num_layers > 1)
        #   batch_first = True so input is [batch, seq_len, d]
        self.lstm = nn.LSTM(
            input_size=d,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Layer normalization on hidden state for training stability
        # (toggleable — sometimes adds little when inputs already z-scored)
        self.layer_norm = nn.LayerNorm(hidden_units) if use_layer_norm else nn.Identity()
        
        # Linear output layer:
        #   maps last hidden state h_last ∈ R^{hidden_units}
        #   to predicted delta or next state ∈ R^{d}
        self.out = nn.Linear(hidden_units, d)
    
    def forward(self, x_seq):
        """
        Forward pass: predict next latent state from sequence.
        
        Parameters
        ----------
        x_seq : torch.Tensor
            Input sequence of latent states.
            Shape: [batch_size, seq_len=lag, d]
        
        Returns
        -------
        y_pred : torch.Tensor
            Predicted next latent state.
            Shape: [batch_size, d]
        
        Notes
        -----
        The LSTM processes the entire sequence and we use the final
        hidden state from the last layer to make the prediction.
        
        LSTM output shapes:
            output: [batch_size, seq_len, hidden_units]
                Hidden states at each time step.
            h_n: [num_layers, batch_size, hidden_units]
                Final hidden state for each layer.
            c_n: [num_layers, batch_size, hidden_units]
                Final cell state for each layer.
        
        We extract h_n[-1] (last layer's final hidden state) and
        pass it through the linear output layer.
        """
        # LSTM output:
        #   output: [batch_size, seq_len, hidden_units]
        #   (h_n, c_n): each [num_layers, batch_size, hidden_units]
        output, (h_n, c_n) = self.lstm(x_seq)
        
        # Take the last layer's hidden state at the final time step:
        # h_n[-1]: [batch_size, hidden_units]
        h_last = h_n[-1]
        
        # Layer normalization for training stability
        h_last = self.layer_norm(h_last)
        
        # Linear map to delta or next latent state:
        # delta_or_pred: [batch_size, d]
        delta_or_pred = self.out(h_last)
        
        if self.residual:
            # Residual connection: y_pred = x_last + delta
            # x_seq[:, -1, :] is the most recent input state
            y_pred = x_seq[:, -1, :] + delta_or_pred
        else:
            y_pred = delta_or_pred
        
        return y_pred
    
    def __repr__(self):
        """String representation of the model."""
        return (
            f"LatentLSTMROM(\n"
            f"  d={self.d},\n"
            f"  hidden_units={self.hidden_units},\n"
            f"  num_layers={self.num_layers},\n"
            f"  dropout={self.dropout_rate},\n"
            f"  residual={self.residual},\n"
            f"  layer_norm={self.use_layer_norm},\n"
            f"  total_params={sum(p.numel() for p in self.parameters())}\n"
            f")"
        )


def train_lstm_rom(X_all, Y_all, config, out_dir, Y_multi=None):
    """
    Train an LSTM ROM on latent sequences.
    
    This function trains a LatentLSTMROM model using windowed latent sequences,
    with train/validation split, early stopping, and model checkpointing.
    
    Key improvements over naive LSTM training:
    - Input z-scoring normalization (per-mode standardization, train-set only)
    - Residual/delta formulation: predict Δy instead of raw y
    - 2-phase cosine+linear scheduled sampling schedule
    - Multi-step rollout loss with real ground-truth targets
    - LayerNorm (toggleable) for training stability
    
    Parameters
    ----------
    X_all : np.ndarray, shape [N_samples, lag, d]
        Input latent sequences (windows).
    Y_all : np.ndarray, shape [N_samples, d]
        One-step-ahead latent targets.
    config : dict or config object
        Must provide rom.models.lstm.* hyperparameters.
    out_dir : str
        Directory where model and logs will be saved.
    Y_multi : np.ndarray or None, shape [N_samples, k_steps, d]
        Multi-step-ahead ground-truth targets. If provided, used for
        supervised multi-step rollout loss. If None, multi-step loss
        uses self-consistency (penalizes drift of model's own rollout).
    
    Returns
    -------
    model_path : str
        Path to the best saved model (state_dict).
    best_val_loss : float
        Best validation loss achieved.
    """
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    # Get shapes
    N_samples, lag, d = X_all.shape
    print(f"\nTraining LSTM ROM:")
    print(f"  Samples: {N_samples:,}")
    print(f"  Lag: {lag}")
    print(f"  Latent dimension: {d}")
    
    # Train/validation split (80/20)
    rng = np.random.RandomState(42)  # Local RNG to avoid polluting global state
    perm = rng.permutation(N_samples)
    N_train = int(0.8 * N_samples)
    train_idx = perm[:N_train]
    val_idx = perm[N_train:]
    
    print(f"\nData split:")
    print(f"  Training samples:   {N_train:,} ({N_train/N_samples*100:.1f}%)")
    print(f"  Validation samples: {N_samples - N_train:,} ({(N_samples-N_train)/N_samples*100:.1f}%)")
    
    # ── Helper to extract config with defaults ──
    def _get(key, default):
        """Extract from lstm_config dict or object with fallback."""
        if hasattr(config, 'rom'):
            return getattr(config.rom.models.lstm, key, default)
        else:
            return config['rom']['models']['lstm'].get(key, default)
    
    if hasattr(config, 'rom'):
        lstm_config = config.rom.models.lstm
        batch_size = lstm_config.batch_size
        hidden_units = lstm_config.hidden_units
        num_layers = lstm_config.num_layers
        learning_rate = lstm_config.learning_rate
        max_epochs = lstm_config.max_epochs
        patience = lstm_config.patience
    else:
        lstm_config = config['rom']['models']['lstm']
        batch_size = lstm_config['batch_size']
        hidden_units = lstm_config['hidden_units']
        num_layers = lstm_config['num_layers']
        learning_rate = lstm_config['learning_rate']
        max_epochs = lstm_config['max_epochs']
        patience = lstm_config['patience']
    
    # Optional hyperparameters with sensible defaults
    weight_decay     = _get('weight_decay', 1e-5)
    gradient_clip    = _get('gradient_clip', 5.0)
    dropout          = _get('dropout', 0.0)
    residual         = _get('residual', True)
    normalize_input  = _get('normalize_input', True)
    use_layer_norm   = _get('use_layer_norm', True)
    
    # Scheduled sampling — 2-phase cosine/linear ramp (professor's suggestion)
    # Phase 1: epochs ss_warmup..ss_phase1_end → ramp 0% → ss_phase1_ratio
    # Phase 2: epochs ss_phase1_end..ss_phase2_end → ramp to ss_max_ratio
    scheduled_sampling = _get('scheduled_sampling', True)
    ss_warmup         = _get('ss_warmup', 20)         # Pure teacher-forcing warmup
    ss_phase1_end     = _get('ss_phase1_end', 200)     # End of gentle ramp
    ss_phase1_ratio   = _get('ss_phase1_ratio', 0.3)   # Ratio at end of phase 1
    ss_phase2_end     = _get('ss_phase2_end', 400)      # End of aggressive ramp
    ss_max_ratio      = _get('ss_max_ratio', 0.5)       # Max self-feeding ratio
    
    # Backward compat: old configs may have ss_start_epoch/ss_end_epoch
    if not hasattr(config, 'rom'):
        if 'ss_start_epoch' in lstm_config and 'ss_warmup' not in lstm_config:
            ss_warmup = lstm_config['ss_start_epoch']
            ss_phase1_end = int((lstm_config['ss_start_epoch'] + lstm_config.get('ss_end_epoch', 500)) / 2)
            ss_phase2_end = lstm_config.get('ss_end_epoch', 500)
    
    # Multi-step rollout loss (professor's 5th suggestion)
    # Loss = (1-α)*L_1step + α*L_kstep, where L_kstep rolls forward k steps
    multistep_loss    = _get('multistep_loss', True)
    multistep_k       = _get('multistep_k', 5)          # Rollout horizon
    multistep_alpha   = _get('multistep_alpha', 0.3)     # Weight of k-step loss
    
    # ── Input normalization (z-scoring) ──
    # POD coefficients can have very different scales across modes.
    # Standardizing to zero-mean, unit-variance helps LSTM training.
    if normalize_input:
        # Compute per-mode statistics from TRAINING set only
        X_flat = X_all[train_idx].reshape(-1, d)  # [N_train*lag, d]
        input_mean = X_flat.mean(axis=0)  # [d]
        input_std_raw = X_flat.std(axis=0)  # [d]
        
        # Clamp std: if a mode has near-zero variance, don't blow up
        STD_FLOOR = 1e-8
        near_zero = input_std_raw < STD_FLOOR
        if near_zero.any():
            n_clamped = near_zero.sum()
            print(f"\n  ⚠ {n_clamped} mode(s) have near-zero std, clamping to {STD_FLOOR}")
        input_std = np.maximum(input_std_raw, STD_FLOOR)  # [d]
        
        # Apply normalization to ALL data (train+val) using train-set stats
        X_all_norm = (X_all - input_mean) / input_std
        Y_all_norm = (Y_all - input_mean) / input_std
        
        # Save normalization stats for inference
        np.savez(os.path.join(out_dir, "lstm_normalization.npz"),
                 input_mean=input_mean, input_std=input_std)
        
        print(f"\n  ✓ Input normalization: per-mode z-scoring (train-set stats only)")
        print(f"    Raw scale:  modes range [{X_all.reshape(-1,d).min():.1f}, {X_all.reshape(-1,d).max():.1f}]")
        print(f"    Per-mode std range: [{input_std.min():.4f}, {input_std.max():.4f}]")
        print(f"    Normalized: modes range [{X_all_norm.reshape(-1,d).min():.2f}, {X_all_norm.reshape(-1,d).max():.2f}]")
    else:
        X_all_norm = X_all
        Y_all_norm = Y_all
        input_mean = np.zeros(d)
        input_std = np.ones(d)
    
    # Create tensors (from normalized data)
    X_train = torch.tensor(X_all_norm[train_idx], dtype=torch.float32)
    Y_train = torch.tensor(Y_all_norm[train_idx], dtype=torch.float32)
    X_val = torch.tensor(X_all_norm[val_idx], dtype=torch.float32)
    Y_val = torch.tensor(Y_all_norm[val_idx], dtype=torch.float32)
    
    # Prepare multi-step targets if provided
    has_multistep_targets = Y_multi is not None and multistep_loss
    if has_multistep_targets:
        # Normalize multi-step targets the same way
        if normalize_input:
            Y_multi_norm = (Y_multi - input_mean) / input_std  # [N, k, d]
        else:
            Y_multi_norm = Y_multi
        Y_multi_train = torch.tensor(Y_multi_norm[train_idx], dtype=torch.float32)
        Y_multi_val = torch.tensor(Y_multi_norm[val_idx], dtype=torch.float32)
        k_avail = Y_multi.shape[1]
        effective_k = min(multistep_k, k_avail)
        print(f"\n  ✓ Multi-step targets: k={effective_k} (ground truth supervised)")
    else:
        Y_multi_train = None
        Y_multi_val = None
        effective_k = multistep_k if multistep_loss else 0
        if multistep_loss and Y_multi is None:
            print(f"\n  ⚠ No Y_multi provided; multi-step loss uses self-consistency")
    
    print(f"\nHyperparameters:")
    print(f"  Batch size:      {batch_size}")
    print(f"  Hidden units:    {hidden_units}")
    print(f"  Num layers:      {num_layers}")
    print(f"  Learning rate:   {learning_rate}")
    print(f"  Weight decay:    {weight_decay}")
    print(f"  Max epochs:      {max_epochs}")
    print(f"  Patience:        {patience}")
    print(f"  Gradient clip:   {gradient_clip}")
    print(f"  Dropout:         {dropout}")
    print(f"  Residual (Δy):   {residual}")
    print(f"  Layer norm:      {use_layer_norm}")
    print(f"  Normalize input: {normalize_input}")
    print(f"  Sched. sampling: {scheduled_sampling}")
    if scheduled_sampling:
        print(f"    Warmup:        0-{ss_warmup} (pure teacher forcing)")
        print(f"    Phase 1:       {ss_warmup}-{ss_phase1_end} → {ss_phase1_ratio:.0%}")
        print(f"    Phase 2:       {ss_phase1_end}-{ss_phase2_end} → {ss_max_ratio:.0%}")
    print(f"  Multistep loss:  {multistep_loss}")
    if multistep_loss:
        print(f"    k-step:        {multistep_k}")
        print(f"    α (weight):    {multistep_alpha}")
    
    # Create DataLoaders (include multi-step targets if available)
    if has_multistep_targets:
        train_ds = TensorDataset(X_train, Y_train, Y_multi_train)
        val_ds = TensorDataset(X_val, Y_val, Y_multi_val)
    else:
        train_ds = TensorDataset(X_train, Y_train)
        val_ds = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    model = LatentLSTMROM(d=d, hidden_units=hidden_units, num_layers=num_layers, 
                          dropout=dropout, residual=residual,
                          use_layer_norm=use_layer_norm)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Samples/param:    {N_samples/total_params:.1f}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler: reduce LR by 0.5 when val_loss plateaus for 20 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
    )
    
    # Training setup
    best_val_loss = float("inf")
    patience_counter = 0
    model_path = os.path.join(out_dir, "lstm_state_dict.pt")
    log_path = os.path.join(out_dir, "training_log.csv")
    
    # Initialize training log
    with open(log_path, 'w') as f:
        f.write("epoch,train_loss,val_loss\n")
    
    print(f"\nStarting training...")
    print(f"{'Epoch':>6s} {'Train Loss':>12s} {'Val Loss':>12s} {'Best':>6s} {'Patience':>8s}")
    print("-" * 60)
    
    # ── Helper: compute 2-phase scheduled sampling ratio ──
    def _ss_ratio(epoch):
        """2-phase SS schedule: warmup → gentle ramp → aggressive ramp."""
        if not scheduled_sampling or epoch < ss_warmup:
            return 0.0
        if epoch < ss_phase1_end:
            # Phase 1: cosine ramp 0 → ss_phase1_ratio
            t = (epoch - ss_warmup) / max(1, ss_phase1_end - ss_warmup)
            return ss_phase1_ratio * 0.5 * (1 - np.cos(np.pi * t))
        elif epoch < ss_phase2_end:
            # Phase 2: linear ramp ss_phase1_ratio → ss_max_ratio
            t = (epoch - ss_phase1_end) / max(1, ss_phase2_end - ss_phase1_end)
            return ss_phase1_ratio + (ss_max_ratio - ss_phase1_ratio) * t
        else:
            return ss_max_ratio
    
    # Training loop
    for epoch in range(max_epochs):
        ss_ratio = _ss_ratio(epoch)
        
        # Training phase
        model.train()
        train_loss_sum = 0.0
        
        for batch in train_loader:
            # Unpack: 2 elements (x, y) or 3 elements (x, y, y_multi)
            if has_multistep_targets:
                x_batch, y_batch, y_multi_batch = batch
                y_multi_batch = y_multi_batch.to(device)
            else:
                x_batch, y_batch = batch
                y_multi_batch = None
            # Move to device
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # ── Scheduled sampling: true multi-step rollout within the window ──
            # Instead of just swapping one token, we roll forward inside the
            # window using the model's own predictions. This directly targets
            # the exposure bias that causes rollout drift.
            if ss_ratio > 0 and x_batch.size(1) >= 2:
                x_batch = x_batch.clone()
                B, L, D = x_batch.shape
                # For each position in the window (starting from position 1),
                # with probability ss_ratio, replace it with model prediction
                # from the previous window state.
                for pos in range(1, L):
                    mask = torch.rand(B, device=device) < ss_ratio  # [B]
                    if mask.any():
                        with torch.no_grad():
                            # Build window ending at pos-1
                            if pos >= L:
                                break
                            # Use the (possibly already corrupted) prefix
                            window_end = pos  # exclusive
                            window_start = max(0, window_end - L)
                            sub_window = x_batch[:, window_start:window_end, :]
                            # Pad if needed to maintain lag length
                            if sub_window.size(1) < L:
                                pad = x_batch[:, :L - sub_window.size(1), :]
                                sub_window = torch.cat([pad, sub_window], dim=1)
                            y_model = model(sub_window)  # [B, D]
                        x_batch[mask, pos, :] = y_model[mask]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # ── Forward pass: 1-step prediction ──
            y_pred = model(x_batch)
            loss_1step = criterion(y_pred, y_batch)
            
            # ── Multi-step rollout loss ──
            # Unroll the model k steps from the input window and penalize
            # accumulated error.  When Y_multi ground-truth targets are
            # available (from build_multistep_latent_dataset) we use
            # *supervised* k-step loss; otherwise fall back to self-
            # consistency (penalise drift between consecutive predictions).
            if multistep_loss and multistep_k > 1 and multistep_alpha > 0:
                rollout_loss = torch.tensor(0.0, device=device)
                window = x_batch  # [B, lag, d]
                prev_pred = y_pred  # step-1 prediction (used only for
                                    # self-consistency fallback)

                for step in range(2, multistep_k + 1):
                    # Shift window: drop oldest, append previous prediction
                    window = torch.cat(
                        [window[:, 1:, :], prev_pred.unsqueeze(1).detach()],
                        dim=1
                    )
                    y_k = model(window)  # [B, d]

                    if y_multi_batch is not None:
                        # Supervised: compare to real future target
                        # y_multi_batch[:, s, :] is target for step s+1
                        # step ranges 2..k  → index = step - 1
                        idx = step - 1  # 0-based into Y_multi's k dim
                        if idx < y_multi_batch.size(1):
                            rollout_loss = rollout_loss + criterion(
                                y_k, y_multi_batch[:, idx, :]
                            )
                        else:
                            # Fallback for steps beyond available targets
                            rollout_loss = rollout_loss + criterion(
                                y_k, prev_pred.detach()
                            )
                    else:
                        # Self-consistency fallback: penalise drift
                        rollout_loss = rollout_loss + criterion(
                            y_k, prev_pred.detach()
                        )
                    prev_pred = y_k

                rollout_loss = rollout_loss / (multistep_k - 1)
                loss = (1 - multistep_alpha) * loss_1step + multistep_alpha * rollout_loss
            else:
                loss = loss_1step
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=gradient_clip
                )
            
            # Optimizer step
            optimizer.step()
            
            # Accumulate training loss (report 1-step for comparability)
            train_loss_sum += loss_1step.item() * x_batch.size(0)
        
        # Compute mean training loss
        train_loss = train_loss_sum / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                if has_multistep_targets:
                    x_batch, y_batch, y_multi_batch = batch
                    y_multi_batch = y_multi_batch.to(device)
                else:
                    x_batch, y_batch = batch
                    y_multi_batch = None
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(x_batch)
                loss_1step = criterion(y_pred, y_batch)

                # Include multi-step rollout loss in val metric for
                # consistency with training loss (helps diagnostics).
                if multistep_loss and multistep_k > 1 and multistep_alpha > 0:
                    rollout_loss = torch.tensor(0.0, device=device)
                    window = x_batch
                    prev_pred = y_pred
                    for step in range(2, multistep_k + 1):
                        window = torch.cat(
                            [window[:, 1:, :], prev_pred.unsqueeze(1)],
                            dim=1
                        )
                        y_k = model(window)
                        if y_multi_batch is not None:
                            idx = step - 1
                            if idx < y_multi_batch.size(1):
                                rollout_loss = rollout_loss + criterion(
                                    y_k, y_multi_batch[:, idx, :]
                                )
                            else:
                                rollout_loss = rollout_loss + criterion(
                                    y_k, prev_pred
                                )
                        else:
                            rollout_loss = rollout_loss + criterion(y_k, prev_pred)
                        prev_pred = y_k
                    rollout_loss = rollout_loss / (multistep_k - 1)
                    loss = (1 - multistep_alpha) * loss_1step + multistep_alpha * rollout_loss
                else:
                    loss = loss_1step
                val_loss_sum += loss.item() * x_batch.size(0)
        
        # Compute mean validation loss
        val_loss = val_loss_sum / len(val_loader.dataset)
        
        # Log to CSV
        with open(log_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.8f},{val_loss:.8f}\n")
        
        # Step the LR scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Early stopping logic
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model state_dict along with model config
            torch.save({
                'state_dict': model.state_dict(),
                'residual': residual,
                'normalize_input': normalize_input,
                'use_layer_norm': use_layer_norm,
                'input_mean': input_mean,
                'input_std': input_std,
                'd': d,
                'hidden_units': hidden_units,
                'num_layers': num_layers,
                'dropout': dropout,
                'lag': lag,
            }, model_path)
            # Also save raw state_dict for backward compatibility
            torch.save(model.state_dict(), os.path.join(out_dir, "lstm_state_dict_raw.pt"))
            best_marker = "  *"
        else:
            patience_counter += 1
            best_marker = ""
        
        # Print progress (show LR and SS ratio if active)
        lr_info = f" lr={current_lr:.1e}" if current_lr < learning_rate else ""
        ss_info = f" ss={ss_ratio:.2f}" if ss_ratio > 0 else ""
        print(f"{epoch+1:6d} {train_loss:12.6f} {val_loss:12.6f} {best_marker:6s} {patience_counter:3d}/{patience:3d}{lr_info}{ss_info}")
        
        # Check early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    print(f"\nTraining complete!")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Model saved to: {model_path}")
    print(f"  Training log: {log_path}")
    
    return model_path, best_val_loss


def forecast_with_lstm(model, y_init_window, n_steps, input_mean=None, input_std=None):
    """
    Roll out the LSTM ROM in latent space for n_steps, in closed loop.
    
    Supports normalization: if input_mean/input_std are provided, the initial
    window is normalized before feeding to the model, and predictions are
    un-normalized before returning. The internal rollout stays in normalized
    space for consistency.
    
    Parameters
    ----------
    model : LatentLSTMROM
        Trained LSTM ROM (PyTorch model).
    y_init_window : np.ndarray, shape [lag, d]
        Initial latent window in ORIGINAL (un-normalized) space.
    n_steps : int
        Number of forecast steps.
    input_mean : np.ndarray or None, shape [d]
        Per-mode mean for z-score normalization.
    input_std : np.ndarray or None, shape [d]
        Per-mode std for z-score normalization.
    
    Returns
    -------
    ys_pred : np.ndarray, shape [n_steps, d]
        Predicted latent states in ORIGINAL space.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Normalize if stats provided
    use_norm = input_mean is not None and input_std is not None
    if use_norm:
        y_init_norm = (y_init_window - input_mean) / input_std
    else:
        y_init_norm = y_init_window
    
    # Convert to tensor: [1, lag, d]
    y_window = torch.tensor(y_init_norm, dtype=torch.float32, device=device).unsqueeze(0)
    
    ys_pred = []
    
    with torch.no_grad():
        for _ in range(n_steps):
            # Predict next state (in normalized space if normalize was used)
            y_next = model(y_window)  # [1, d]
            
            # Store prediction (un-normalize for output)
            y_next_np = y_next.cpu().numpy()[0]  # [d], in normalized space
            if use_norm:
                ys_pred.append(y_next_np * input_std + input_mean)
            else:
                ys_pred.append(y_next_np)
            
            # Update window (stay in normalized space for next step)
            y_window = torch.cat(
                [y_window[:, 1:, :], y_next.unsqueeze(1)],
                dim=1
            )
    
    return np.stack(ys_pred, axis=0)


def lstm_forecast_fn_factory(model, input_mean=None, input_std=None):
    """
    Create a forecast function closure for a specific LSTM model.
    
    This factory function returns a callable that can be used with the
    generic ROM evaluation pipeline. The returned function has the same
    signature as MVAR forecast functions, making LSTM and MVAR interchangeable
    in the evaluation code.
    
    Parameters
    ----------
    model : LatentLSTMROM
        Trained LSTM ROM model.
    input_mean : np.ndarray or None, shape [d]
        Per-mode mean for z-score normalization.
    input_std : np.ndarray or None, shape [d]
        Per-mode std for z-score normalization.
    
    Returns
    -------
    forecast_fn : callable
        A function with signature:
            forecast_fn(y_init_window, n_steps) -> ys_pred
    """
    def _forecast_fn(y_init_window, n_steps):
        """Forecast wrapper that calls forecast_with_lstm with bound model."""
        return forecast_with_lstm(model, y_init_window, n_steps,
                                  input_mean=input_mean, input_std=input_std)
    
    return _forecast_fn


def load_lstm_model(model_dir, device='cpu'):
    """
    Load a trained LSTM model with all its metadata (normalization, config).
    
    Supports both new checkpoint format (with metadata) and legacy format
    (raw state_dict only).
    
    Parameters
    ----------
    model_dir : str or Path
        Directory containing lstm_state_dict.pt and optionally lstm_normalization.npz
    device : str
        Device to load model onto.
    
    Returns
    -------
    model : LatentLSTMROM
        Loaded model in eval mode.
    input_mean : np.ndarray or None
        Normalization mean (None if not used).
    input_std : np.ndarray or None
        Normalization std (None if not used).
    """
    from pathlib import Path
    model_dir = Path(model_dir)
    # Accept either a directory or a direct path to the checkpoint file
    if model_dir.is_file():
        model_path = model_dir
    else:
        model_path = model_dir / "lstm_state_dict.pt"
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # New format: checkpoint is a dict with metadata
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        d = checkpoint['d']
        hidden_units = checkpoint['hidden_units']
        num_layers = checkpoint['num_layers']
        dropout = checkpoint.get('dropout', 0.0)
        residual = checkpoint.get('residual', False)
        use_layer_norm = checkpoint.get('use_layer_norm', True)
        
        model = LatentLSTMROM(d=d, hidden_units=hidden_units, 
                              num_layers=num_layers, dropout=dropout,
                              residual=residual, use_layer_norm=use_layer_norm)
        model.load_state_dict(checkpoint['state_dict'])
        
        if checkpoint.get('normalize_input', False):
            input_mean = checkpoint['input_mean']
            input_std = checkpoint['input_std']
        else:
            input_mean, input_std = None, None
    else:
        # Legacy format: raw state_dict, need external info
        # Try to load normalization from separate file
        input_mean, input_std = None, None
        norm_file = model_dir / "lstm_normalization.npz"
        if norm_file.exists():
            norm_data = np.load(norm_file)
            input_mean = norm_data['input_mean']
            input_std = norm_data['input_std']
        
        # Can't infer model architecture from raw state_dict alone
        # Caller must provide d, hidden_units, etc.
        # Return the state_dict for manual loading
        raise ValueError(
            "Legacy state_dict format detected. Use load_lstm_model_legacy() "
            "or retrain with the new format."
        )
    
    model.to(device)
    model.eval()
    return model, input_mean, input_std


if __name__ == "__main__":
    """
    Basic validation tests for the LatentLSTMROM model.
    """
    print("\n" + "="*80)
    print("Testing LatentLSTMROM Model Architecture")
    print("="*80 + "\n")
    
    # Test 1: Basic instantiation and forward pass
    print("Test 1: Basic model instantiation")
    print("-" * 80)
    d = 25
    hidden_units = 16
    num_layers = 1
    
    model = LatentLSTMROM(d=d, hidden_units=hidden_units, num_layers=num_layers)
    print(f"Created model:\n{model}\n")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    lstm_params = sum(p.numel() for p in model.lstm.parameters())
    linear_params = sum(p.numel() for p in model.out.parameters())
    
    print(f"Parameter breakdown:")
    print(f"  LSTM parameters:   {lstm_params:,}")
    print(f"  Linear parameters: {linear_params:,}")
    print(f"  Total parameters:  {total_params:,}\n")
    
    # Test 2: Forward pass with single sample
    print("Test 2: Forward pass with single sample")
    print("-" * 80)
    lag = 10
    batch_size = 1
    
    x_single = torch.randn(batch_size, lag, d)
    y_pred = model(x_single)
    
    print(f"Input shape:  {x_single.shape}  [batch, lag, d]")
    print(f"Output shape: {y_pred.shape}  [batch, d]")
    print(f"✓ Single sample forward pass successful\n")
    
    # Test 3: Forward pass with batch
    print("Test 3: Forward pass with batch")
    print("-" * 80)
    batch_size = 32
    
    x_batch = torch.randn(batch_size, lag, d)
    y_pred_batch = model(x_batch)
    
    print(f"Input shape:  {x_batch.shape}  [batch, lag, d]")
    print(f"Output shape: {y_pred_batch.shape}  [batch, d]")
    print(f"✓ Batch forward pass successful\n")
    
    # Test 4: Multi-layer LSTM
    print("Test 4: Multi-layer LSTM (2 layers)")
    print("-" * 80)
    model_deep = LatentLSTMROM(d=d, hidden_units=32, num_layers=2)
    print(f"Created deep model:\n{model_deep}\n")
    
    y_pred_deep = model_deep(x_batch)
    print(f"Input shape:  {x_batch.shape}  [batch, lag, d]")
    print(f"Output shape: {y_pred_deep.shape}  [batch, d]")
    print(f"✓ Multi-layer forward pass successful\n")
    
    # Test 5: Different lag values
    print("Test 5: Variable sequence lengths")
    print("-" * 80)
    for test_lag in [5, 10, 20]:
        x_test = torch.randn(16, test_lag, d)
        y_test = model(x_test)
        print(f"  lag={test_lag:2d}: input {x_test.shape} → output {y_test.shape}")
    print(f"✓ Variable lag handling successful\n")
    
    # Test 6: Gradient flow check
    print("Test 6: Gradient flow check")
    print("-" * 80)
    x_grad = torch.randn(8, 10, d, requires_grad=True)
    y_grad = model(x_grad)
    loss = y_grad.sum()
    loss.backward()
    
    has_grad = x_grad.grad is not None
    grad_norm = x_grad.grad.norm().item() if has_grad else 0.0
    
    print(f"Input requires_grad: {x_grad.requires_grad}")
    print(f"Output requires_grad: {y_grad.requires_grad}")
    print(f"Gradient computed: {has_grad}")
    print(f"Gradient norm: {grad_norm:.6f}")
    print(f"✓ Gradient flow successful\n")
    
    # Test 7: Model state dict
    print("Test 7: Model state dict inspection")
    print("-" * 80)
    state_dict = model.state_dict()
    print(f"Model state dict keys:")
    for key in state_dict.keys():
        shape = state_dict[key].shape
        print(f"  {key:30s}: {tuple(shape)}")
    print(f"✓ State dict accessible\n")
    
    print("="*80)
    print("✅ All tests passed! LatentLSTMROM is ready for training.")
    print("="*80)
    print("\nNext steps:")
    print("  - PART 4: Implement training loop")
    print("  - PART 5: Implement closed-loop forecasting")
    print()
