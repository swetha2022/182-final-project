import torch

class MuonOptimizerWrapper:
    """
    Wrapper for Muon optimizer that flattens 4D parameters to 2D.
    For 4D tensors (e.g., conv weights [C_out, C_in, H, W]), flattens to 2D [C_out*H*W, C_in].
    For other tensors, keeps them as-is.
    """
    def __init__(self, original_params, lr, weight_decay):
        self.original_params = list(original_params)
        self.conv_params = []
        self.flattened_params = []
        self.matrix_params = []
        self.vector_params = []
        
        for param in self.original_params:
            if param.dim() == 4:
                flattened_view = param.view(-1, param.shape[1])
                flattened_param = torch.nn.Parameter(flattened_view.clone())
                self.flattened_params.append(flattened_param)
                self.conv_params.append(param)
            elif param.dim() == 2:
                self.matrix_params.append(param)
            elif param.dim() == 1:
                self.vector_params.append(param)
            else:
                raise Exception("Non 4d, 2d, or 1d parameter")
        
        self.vector_optimizer = torch.optim.AdamW(self.vector_params, lr=lr, weight_decay=weight_decay)
        self.optimizer = torch.optim.Muon(self.flattened_params + self.matrix_params, lr=lr, weight_decay=weight_decay)
    
    def zero_grad(self):
        for param in self.original_params:
            if param.grad is not None:
                param.grad.zero_()
        self.optimizer.zero_grad()
        self.vector_optimizer.zero_grad()
    
    def step(self):
        # Before step, sync gradients from original to flattened params
        for orig_param, flat_param in zip(self.conv_params, self.flattened_params):
            flat_param.grad = orig_param.grad.view(-1, orig_param.shape[1])
        
        # Perform optimizer step
        self.optimizer.step()
        self.vector_optimizer.step()
        
        # After step, sync updated parameters back from flattened to original
        for orig_param, flat_param in zip(self.conv_params, self.flattened_params):
            orig_param.data.copy_(flat_param.data.view(orig_param.shape))
