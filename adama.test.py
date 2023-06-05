import torch
import unittest
from adama import AdamA

class TestAdamA(unittest.TestCase):
    def test_step(self):
        # Define a simple model and loss function
        model = torch.nn.Linear(2, 1)
        loss_fn = torch.nn.MSELoss()

        # Define the optimizer
        optimizer = AdamA(model.parameters())

        # Define the input and target tensors
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = torch.tensor([[3.0], [7.0]])

        # Perform a single optimization step
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        # Check that the model parameters have been updated
        for param in model.parameters():
            self.assertNotEqual(param.grad, None)
            self.assertNotEqual(param.data, None)
            self.assertNotEqual(param.grad.data, None)
            self.assertNotEqual(param.data.grad, None)

    def test_invalid_parameters(self):
        # Test that the optimizer raises an error for invalid parameters
        with self.assertRaises(ValueError):
            optimizer = AdamA([], lr=-1.0)
        with self.assertRaises(ValueError):
            optimizer = AdamA([], eps=-1.0)
        with self.assertRaises(ValueError):
            optimizer = AdamA([], betas=(1.0, 0.9))
        with self.assertRaises(ValueError):
            optimizer = AdamA([], betas=(0.9, 1.0))
        with self.assertRaises(ValueError):
            optimizer = AdamA([], weight_decay=-1.0)

if __name__ == '__main__':
    unittest.main()