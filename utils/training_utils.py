import torch.optim as optim
import torch.nn as nn

# baseline loss function
criterion = nn.CrossEntropyLoss()

def training_loop(model_package,lambda_reg,inputs,labels) -> "loss":
    """Runs a single training loop for a single model

    Args:
        model_package (list(dict)): list of dictionaries containing necessary elements of regularizers for training
        lambda_reg (float): modulation value to scale the impact of regularization terms
        inputs (torch.Tensor): data to be predicted upon
        labels (torch.Tensor): actual outputs

    Returns:
        loss (float): loss value
    """

    model = model_package["model"]
    regularizer_func = model_package["regularizer_func"]
    optimizer = model_package["optimizer"]

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward pass
    outputs = model(inputs)

    # calculate regularization values for all nodes
    regularization_values = [regularizer_func(params) for type, params in model.named_parameters() if type.endswith('weight')]
    
    # calculate baseline loss + modulated regularization value
    loss = criterion(outputs, labels) + lambda_reg * sum(regularization_values)

    # calculate gradients with respect to loss
    loss.backward()

    # apply gradients to parameters
    optimizer.step()

    # return loss value for analysis
    return loss.item()