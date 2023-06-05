import math
import torch


class AdamA(torch.optim.Optimizer):
    """Implements the Adam with Accumulate algorithm laid out in the paper
    "Adam Accumulation to Reduce Memory Footprints of both Activations and Gradients for Large-scale DNN Training"
    by Yijia Zhang, Yibo Han, Shijie Cao, Guohao Dai, Youshan Miao, Ting Cao, Fan Yang, and Ningyi Xu

    https://arxiv.org/abs/2305.19982
    """

    def __init__(
        self,
        params,
        grad_accumulate_step=1,
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super(AdamA, self).__init__(params, defaults)
        self.microstep = 1
        self.micro_accumulation_step = grad_accumulate_step

    def __setstate__(self, state):
        """
        Set the optimizer state from a given state dictionary.

        Args:
            state (dict): State dictionary containing optimizer state information.
        """
        super(AdamA, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None, first_block=True, parameters_to_update=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
            first_block (bool, optional): Whether to perform the first block of the algorithm.
            param_group_idx_container (list, optional): A list of indices of parameter groups to update.

        Returns:
            None
        """

        if parameters_to_update is None:
            return self.step(closure=closure)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        for parameter_group in group["params"]:
            parameter = parameter_group[parameters_to_update]
            if parameter.grad is None:
                continue
            gradient = parameter.grad.data
            if gradient.is_sparse:
                raise RuntimeError("Adam does not support sparse gradients")

            amsgrad = group["amsgrad"]

            state = self.state[parameter]

            if len(state) == 0:
                state["step"] = 0
                state["ema"] = torch.zeros_like(  # EMA of the gradients
                    parameter.data, memory_format=torch.preserve_format
                )

                state["squared_ema"] = torch.zeros_like(  # Squared EMA of gradients
                    parameter.data, memory_format=torch.preserve_format
                )
                if amsgrad:
                    state[
                        "max_squared_ema"
                    ] = torch.zeros_like(  # Maximum of all squared EMA of gradients
                        parameter.data, memory_format=torch.preserve_format
                    )
            # Fetch the EMA and squared EMA of gradients for the current parameter group
            squared_ema, ema = state["squared_ema"], state["ema"]
            if amsgrad:
                max_squared_ema = state["max_squared_ema"]
            beta_a, beta_b = group["betas"]

            # Now we perform microbatching. If the current microbatch is the first microbatch of the macrobatch, we initialize the EMA and squared EMA of gradients with the current microbatch of gradients. Otherwise, we update the EMA and squared EMA of gradients with the current microbatch of gradients.

            if self.microstep == 1:
                """First microstep, initialize the EMA and squared EMA of gradients with the current microbatch of gradients."""
                ema.mul_(beta_a).add_(gradient, alpha=1 - beta_a)
                squared_ema.mul_(beta_b).addcmul_(gradient, gradient, value=1 - beta_b)

            elif self.microstep == self.micro_accumulation_step:  
                """This microstep is the last of the microbatch. Consequently, we update the parameters"""
                
                ema.add_(gradient, alpha=1 - beta_a)
                squared_ema.addcmul_(gradient, gradient, value=1 - beta_b)
                state["step"] += 1

                # The bias correction is applied to the EMA and squared EMA of gradients, and the maximum of all squared EMA of gradients if AMSGrad is used.
                bias_correction_a = 1 - beta_a ** state["step"]
                bias_correction_b = 1 - beta_b ** state["step"]

                # If AMSGrad is used, we update the maximum of all squared EMA of gradients. Otherwise, we update the EMA of gradients.
                to_update = torch.max(max_squared_ema, squared_ema) ** .5 if amsgrad else squared_ema
                
                # Denominator of the update rule. Epsilon is added to improve numerical stability.
                denom = (to_update / (bias_correction_b**0.5)).add_(group["eps"])
                    

                # The step size is computed.
                step_size = group["lr"] / bias_correction_a

                # Then we update the parameters.
                parameter.data.addcdiv_(ema, denom, value=-step_size)
            else:
                """ This step is between the other two, so we update the EMA and squared EMA of gradients."""
                ema.add_(gradient, alpha=1-beta_a)
                squared_ema.addcmul_(gradient, gradient, value=1 - beta_b)

        if first_block:
            if self.microstep == self.micro_accumulation_step:
                self.microstep = 1
            else:
                self.microstep += 1
        return loss