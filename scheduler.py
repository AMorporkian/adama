from typing import Callable, Optional

import torch
import pprint

from adama import AdamA


class MicroGradScheduler:
    """
    A class that schedules micro steps for a given model and optimizer.

    Attributes:
    -----------
    parameters : torch.Tensor
        The tensor of parameters to be optimized.
    optimizer : torch.optim.Optimizer
        The optimizer to be used.
    layer_map : dict
        A dictionary that maps each layer to its next layer.
    hook_handlers : list
        A list of hook handlers.
    param_index_map : dict
        A dictionary that maps each parameter to its index.
    global_step : int
        The current global step.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        parameters: torch.Tensor,
        optimizer: AdamA
        ):
        """
        Initializes the MicroGradScheduler.

        Parameters:
        -----------
        parameters : torch.Tensor
            The tensor of parameters to be optimized.
        optimizer : torch.optim.Optimizer
            The optimizer to be used.
        layer_map : dict
            A dictionary that maps each layer to its next layer.
        """
        self.model = model
        self.parameters = parameters
        self.optimizer = optimizer
        self.layer_map = {}
        self.hook_handlers = []
        self.param_index_map = {}
        self.global_step = 0
        for index, param in enumerate(self.parameters):
            self.param_index_map[param] = index
        for layer, next_layer in zip(self.model, self.model[1:]):
            self.layer_map[layer] = next_layer
        self._register_layer_hooks()

    def _register_layer_hooks(self):
        """
        Registers hooks for each layer.
        """
        for layer in self.layer_map.keys():
            hook = layer.register_backward_hook(self._layer_hook_factory())
            self.hook_handlers.append(hook)

    def _layer_hook_factory(self):
        """
        Returns a hook for a layer.
        """

        def layer_hook(module, grad_input, grad_output):
            next_layer = self.layer_map[module]
            for p in next_layer:
                param_idx = self.param_index_map[p]
                self.optimizer.step(None, self.global_step, param_group_idx=self.param_index_map[param_idx])
                if hasattr(p, "grad"):
                    del p.grad

        return layer_hook

    def step(self, closure: Callable = None):
        """
        Performs a micro step.
        """
        for p in self.parameters:
            parameters = self.param_index_map[p]
            self.optimizer.step(
                closure, self.global_step, parameters_to_update=parameters
            )
            if hasattr(p, "grad"):
                del p.grad
        self.global_step += 1

    def get_global_step(self, global_step: int):
        """
        Sets the global step.

        Parameters:
        -----------
        global_step : int
            The current global step.
        """
        self.global_step = global_step

    def map_params_to_indices(self):
        """
        Maps each parameter to its index.
        """
        for idx, p in enumerate(self.parameters):
            self.param_index_map[p] = idx

    def get_next_layer_from_param(self, param: torch.Tensor):
        """
        Returns the next layer from a given parameter.

        Parameters:
        -----------
        param : torch.Tensor
            The parameter to be used.

        Returns:
        --------
        torch.nn.Module
            The next layer.
        """
        return self.layer_map[param]
    
