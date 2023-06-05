import torch
from adama import AdamA
from scheduler import MicroGradScheduler


def test_scheduler():
    # Define a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1),
        torch.nn.Sigmoid()
    )

    # Define some input data
    x = torch.randn(1, 10)
    y = torch.tensor([[0.5]])

    # Define the optimizer
    optimizer = AdamA(model.parameters(), lr=0.1)

    # Create the scheduler
    scheduler = MicroGradScheduler(model, model.parameters(), optimizer)

    # Test the step function
    scheduler.step()
    assert scheduler.global_step == 1

    # Test the get_global_step function
    scheduler.get_global_step(10)
    assert scheduler.global_step == 10

    # Test the map_params_to_indices function
    scheduler.map_params_to_indices()
    assert len(scheduler.param_index_map) == len(list(model.parameters()))
    


    
if __name__ == "__main__":
    test_scheduler()
    