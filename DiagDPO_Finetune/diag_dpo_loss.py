import torch
import torch.nn.functional as F

def compute_time_aware_diag_dpo_loss(
    model_pred_w, 
    model_pred_l, 
    target, 
    timesteps, 
    delta_E, 
    delta_A, 
    delta_I, 
    eta=0.5
):
    """
    Computes the Time-Aware Diagnostic DPO (Diag-DPO) loss for Flow Matching models like FLUX.
    
    Args:
        model_pred_w: The model's flow prediction for the winner image (batch_size, ...)
        model_pred_l: The model's flow prediction for the loser image (batch_size, ...)
        target: The target flow (batch_size, ...)
        timesteps: The current timesteps (batch_size,) where 1.0 is pure noise and 0.0 is clear image
        delta_E: Difference in Existence score (winner - loser) (batch_size,)
        delta_A: Difference in Appearance score (winner - loser) (batch_size,)
        delta_I: Difference in Interaction score (winner - loser) (batch_size,)
        eta: Degeneration parameter. If eta=0, it falls back to standard Flow-DPO.
        
    Returns:
        The scalar loss value.
    """
    # Flatten the spatial dimensions to compute MSE per sample
    batch_size = model_pred_w.shape[0]
    pred_w_flat = model_pred_w.view(batch_size, -1)
    pred_l_flat = model_pred_l.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)
    
    # 1. Compute Base Flow Prediction Errors
    error_w = F.mse_loss(pred_w_flat, target_flat, reduction='none').mean(dim=1)
    error_l = F.mse_loss(pred_l_flat, target_flat, reduction='none').mean(dim=1)
    
    # 2. Compute Margin (Standard DPO part)
    margin = error_l - error_w
    base_loss = - F.logsigmoid(margin)
    
    if eta == 0.0:
        return base_loss.mean()
        
    # 3. Dynamic Routing (Alpha): Reward only the dimensions where the winner actually won
    relu_E = F.relu(delta_E)
    relu_A = F.relu(delta_A)
    relu_I = F.relu(delta_I)
    
    sum_relu = relu_E + relu_A + relu_I + 1e-8
    alpha_E = relu_E / sum_relu
    alpha_A = relu_A / sum_relu
    alpha_I = relu_I / sum_relu
    
    # 4. Frequency Separation (Time Kernels)
    sigma = 0.15
    k_E = torch.exp(- ((timesteps - 0.8) ** 2) / (2 * sigma**2))
    k_I = torch.exp(- ((timesteps - 0.5) ** 2) / (2 * sigma**2))
    k_A = torch.exp(- ((timesteps - 0.2) ** 2) / (2 * sigma**2))
    
    # 5. Time-Modulated Weights
    lambda_t = alpha_E * k_E + alpha_I * k_I + alpha_A * k_A
    
    # 6. Degeneration Protection
    weight_t = (1 - eta) + eta * lambda_t
    
    # 7. Final Loss
    final_loss = weight_t * base_loss
    
    return final_loss.mean()
