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
    
    def __init__(self, d, hidden_units=16, num_layers=1):
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
        """
        super().__init__()
        self.d = d
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        
        # LSTM block:
        #   input_size  = d  (latent dimension)
        #   hidden_size = hidden_units
        #   num_layers  = num_layers
        #   batch_first = True so input is [batch, seq_len, d]
        self.lstm = nn.LSTM(
            input_size=d,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Linear output layer:
        #   maps last hidden state h_last ∈ R^{hidden_units}
        #   to next latent state y_pred ∈ R^{d}
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
        
        # Linear map to next latent state:
        # y_pred: [batch_size, d]
        y_pred = self.out(h_last)
        return y_pred
    
    def __repr__(self):
        """String representation of the model."""
        return (
            f"LatentLSTMROM(\n"
            f"  d={self.d},\n"
            f"  hidden_units={self.hidden_units},\n"
            f"  num_layers={self.num_layers},\n"
            f"  total_params={sum(p.numel() for p in self.parameters())}\n"
            f")"
        )


def train_lstm_rom(X_all, Y_all, config, out_dir):
    """
    Train an LSTM ROM on latent sequences.
    
    This function trains a LatentLSTMROM model using windowed latent sequences,
    with train/validation split, early stopping, and model checkpointing.
    
    Parameters
    ----------
    X_all : np.ndarray, shape [N_samples, lag, d]
        Input latent sequences (windows).
    Y_all : np.ndarray, shape [N_samples, d]
        One-step-ahead latent targets.
    config : dict or config object
        Must provide rom.models.lstm.* hyperparameters:
            - batch_size: int
            - hidden_units: int
            - num_layers: int
            - learning_rate: float
            - weight_decay: float
            - max_epochs: int
            - patience: int
            - gradient_clip: float
    out_dir : str
        Directory where model and logs will be saved.
    
    Returns
    -------
    model_path : str
        Path to the best saved model (state_dict).
    best_val_loss : float
        Best validation loss achieved.
    
    Notes
    -----
    The function performs the following steps:
    1. Creates output directory
    2. Splits data into 80% train, 20% validation
    3. Creates PyTorch DataLoaders with specified batch size
    4. Initializes LSTM model and moves to GPU if available
    5. Trains with Adam optimizer and MSE loss
    6. Implements early stopping based on validation loss
    7. Saves best model checkpoint
    8. Writes training log to CSV
    
    Examples
    --------
    >>> X_all = np.random.randn(1000, 10, 25)
    >>> Y_all = np.random.randn(1000, 25)
    >>> model_path, val_loss = train_lstm_rom(X_all, Y_all, config, 'output/lstm')
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
    np.random.seed(42)  # For reproducibility
    perm = np.random.permutation(N_samples)
    N_train = int(0.8 * N_samples)
    train_idx = perm[:N_train]
    val_idx = perm[N_train:]
    
    print(f"\nData split:")
    print(f"  Training samples:   {N_train:,} ({N_train/N_samples*100:.1f}%)")
    print(f"  Validation samples: {N_samples - N_train:,} ({(N_samples-N_train)/N_samples*100:.1f}%)")
    
    # Create tensors
    X_train = torch.tensor(X_all[train_idx], dtype=torch.float32)
    Y_train = torch.tensor(Y_all[train_idx], dtype=torch.float32)
    X_val = torch.tensor(X_all[val_idx], dtype=torch.float32)
    Y_val = torch.tensor(Y_all[val_idx], dtype=torch.float32)
    
    # Extract hyperparameters from config
    # Support both dict-style and object-style access
    if hasattr(config, 'rom'):
        lstm_config = config.rom.models.lstm
        batch_size = lstm_config.batch_size
        hidden_units = lstm_config.hidden_units
        num_layers = lstm_config.num_layers
        learning_rate = lstm_config.learning_rate
        weight_decay = getattr(lstm_config, 'weight_decay', 1e-5)
        max_epochs = lstm_config.max_epochs
        patience = lstm_config.patience
        gradient_clip = getattr(lstm_config, 'gradient_clip', 5.0)
    else:
        lstm_config = config['rom']['models']['lstm']
        batch_size = lstm_config['batch_size']
        hidden_units = lstm_config['hidden_units']
        num_layers = lstm_config['num_layers']
        learning_rate = lstm_config['learning_rate']
        weight_decay = lstm_config.get('weight_decay', 1e-5)
        max_epochs = lstm_config['max_epochs']
        patience = lstm_config['patience']
        gradient_clip = lstm_config.get('gradient_clip', 5.0)
    
    print(f"\nHyperparameters:")
    print(f"  Batch size:      {batch_size}")
    print(f"  Hidden units:    {hidden_units}")
    print(f"  Num layers:      {num_layers}")
    print(f"  Learning rate:   {learning_rate}")
    print(f"  Weight decay:    {weight_decay}")
    print(f"  Max epochs:      {max_epochs}")
    print(f"  Patience:        {patience}")
    print(f"  Gradient clip:   {gradient_clip}")
    
    # Create DataLoaders
    train_ds = TensorDataset(X_train, Y_train)
    val_ds = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    model = LatentLSTMROM(d=d, hidden_units=hidden_units, num_layers=num_layers)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
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
    
    # Training loop
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss_sum = 0.0
        
        for x_batch, y_batch in train_loader:
            # Move to device
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(x_batch)
            
            # Compute loss
            loss = criterion(y_pred, y_batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=gradient_clip
                )
            
            # Optimizer step
            optimizer.step()
            
            # Accumulate training loss
            train_loss_sum += loss.item() * x_batch.size(0)
        
        # Compute mean training loss
        train_loss = train_loss_sum / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                val_loss_sum += loss.item() * x_batch.size(0)
        
        # Compute mean validation loss
        val_loss = val_loss_sum / len(val_loader.dataset)
        
        # Log to CSV
        with open(log_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.8f},{val_loss:.8f}\n")
        
        # Early stopping logic
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model state_dict
            torch.save(model.state_dict(), model_path)
            best_marker = "  *"
        else:
            patience_counter += 1
            best_marker = ""
        
        # Print progress
        print(f"{epoch+1:6d} {train_loss:12.6f} {val_loss:12.6f} {best_marker:6s} {patience_counter:3d}/{patience:3d}")
        
        # Check early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    print(f"\nTraining complete!")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Model saved to: {model_path}")
    print(f"  Training log: {log_path}")
    
    return model_path, best_val_loss


def forecast_with_lstm(model, y_init_window, n_steps):
    """
    Roll out the LSTM ROM in latent space for n_steps, in closed loop.
    
    This function performs autoregressive (closed-loop) forecasting in the
    latent space. At each step, it predicts the next latent state, then uses
    that prediction as input for the subsequent prediction.
    
    Parameters
    ----------
    model : LatentLSTMROM
        Trained LSTM ROM (PyTorch model).
        Should already be on the correct device (CPU or GPU).
    y_init_window : np.ndarray, shape [lag, d]
        Initial latent window (warm-up) for times t_{k-lag}, ..., t_{k-1}.
        Typically obtained from truth: y(t_j) = R(x_truth(t_j)).
        This provides the initial sequence the LSTM needs to start forecasting.
    n_steps : int
        Number of forecast steps to perform.
        Will predict latent states at times t_k, t_{k+1}, ..., t_{k+n_steps-1}.
    
    Returns
    -------
    ys_pred : np.ndarray, shape [n_steps, d]
        Predicted latent states in sequence.
        Each row ys_pred[i] is the predicted latent state at time t_{k+i}.
    
    Notes
    -----
    Closed-loop forecasting means that predictions are fed back as inputs:
    - Initial window: y_{k-lag}, ..., y_{k-1} (from truth)
    - Step 1: Predict y_k using initial window
    - Step 2: Predict y_{k+1} using [y_{k-lag+1}, ..., y_{k-1}, y_k]
    - Step 3: Predict y_{k+2} using [y_{k-lag+2}, ..., y_{k-1}, y_k, y_{k+1}]
    - And so on...
    
    This is different from open-loop (teacher forcing) where we would use
    true values as inputs at each step. Closed-loop is the realistic setting
    for ROM forecasting where we don't have access to future ground truth.
    
    Examples
    --------
    >>> model = LatentLSTMROM(d=25, hidden_units=64, num_layers=2)
    >>> model.load_state_dict(torch.load('lstm_state_dict.pt'))
    >>> model.eval()
    >>> 
    >>> # Initial window from truth (e.g., first 20 timesteps)
    >>> y_init = np.random.randn(20, 25)  # [lag=20, d=25]
    >>> 
    >>> # Forecast 100 steps ahead
    >>> y_forecast = forecast_with_lstm(model, y_init, n_steps=100)
    >>> y_forecast.shape
    (100, 25)
    """
    # Put model in evaluation mode (disables dropout, etc.)
    model.eval()
    
    # Get device from model parameters
    device = next(model.parameters()).device
    
    # Convert initial window to tensor and add batch dimension
    # y_init_window: [lag, d] -> [1, lag, d] for batch_first=True
    y_window = torch.tensor(y_init_window, dtype=torch.float32, device=device)
    y_window = y_window.unsqueeze(0)  # [1, lag, d]
    
    # Initialize list to store predictions
    ys_pred = []
    
    # Perform closed-loop forecasting
    with torch.no_grad():
        for _ in range(n_steps):
            # Predict next latent state: [1, d]
            y_next = model(y_window)
            
            # Save prediction as numpy array: [d]
            ys_pred.append(y_next.cpu().numpy()[0])
            
            # Update window: drop oldest timestep, append new prediction
            # y_window[:, 1:, :] removes first timestep -> [1, lag-1, d]
            # y_next.unsqueeze(1) adds time dimension -> [1, 1, d]
            # Concatenation produces -> [1, lag, d]
            y_window = torch.cat(
                [y_window[:, 1:, :], y_next.unsqueeze(1)],
                dim=1
            )
    
    # Stack all predictions into array: [n_steps, d]
    ys_pred = np.stack(ys_pred, axis=0)
    
    return ys_pred


def lstm_forecast_fn_factory(model):
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
        Should already be loaded with trained weights and moved to device.
    
    Returns
    -------
    forecast_fn : callable
        A function with signature:
            forecast_fn(y_init_window, n_steps) -> ys_pred
        where:
            y_init_window : np.ndarray [lag, d]
            n_steps : int
            ys_pred : np.ndarray [n_steps, d]
    
    Notes
    -----
    This factory pattern allows the evaluation code to be model-agnostic.
    The same evaluation function can work with both MVAR and LSTM by
    simply passing different forecast functions.
    
    Examples
    --------
    >>> # Load trained LSTM model
    >>> model = LatentLSTMROM(d=25, hidden_units=64, num_layers=2)
    >>> model.load_state_dict(torch.load('best_lstm.pt'))
    >>> model.eval()
    >>> model.to('cuda')
    >>> 
    >>> # Create forecast function
    >>> lstm_forecast_fn = lstm_forecast_fn_factory(model)
    >>> 
    >>> # Use in evaluation (same interface as MVAR)
    >>> y_init = np.random.randn(20, 25)
    >>> y_pred = lstm_forecast_fn(y_init, n_steps=100)
    >>> 
    >>> # Can be passed to generic evaluation function
    >>> evaluate_rom(
    ...     model_name="LSTM",
    ...     forecast_next_latent_sequence_fn=lstm_forecast_fn,
    ...     config=config,
    ...     R=R, L=L,
    ...     test_trajectories=test_data,
    ...     out_dir="results/run_001/LSTM"
    ... )
    """
    def _forecast_fn(y_init_window, n_steps):
        """Forecast wrapper that calls forecast_with_lstm with bound model."""
        return forecast_with_lstm(model, y_init_window, n_steps)
    
    return _forecast_fn


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
