import numpy as np
from .integrate import integrate_model_training, rk4

def train_step(dataloader, model, loss_fn, optimizer, print_stat=True):
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    model.train() #Tell model to run in train mode.
    
    for batch, (x, y) in enumerate(dataloader):
        
        pred = model(x)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad() # Reset the stored intermediate gradients
        loss.backward()
        optimizer.step()
        
        if 5 * batch % num_batches == 0 and print_stat:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
    
    return 
    

def test(dataloader, model, loss_fn, ln="\n", print_loss=True):
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    
    model.eval()
    
    test_loss = 0.0

    for x, y in dataloader:

        pred = model(x)

        test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    
    if print_loss:
        print(f"Avg loss: {test_loss:>8f}", end=ln)
    
    return test_loss


def train_loop(train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs, lr=None):

    tmp_lr = optimizer.param_groups[0]['lr']
    if lr is not None:
        optimizer.param_groups[0]['lr'] = lr

    losses = []

    for t in range(1,epochs+1):
        train_step(train_dataloader, model, loss_fn, optimizer, print_stat=False)

        loss = test(test_dataloader, model, loss_fn, print_loss=False)
        losses.append(loss)

        if 100*t % (10*epochs) == 0:
            print(f"Epoch {t:>3d} / {epochs} --------- ", end="")
            print(f"Avg loss: {loss:>8f}")
            
    print("\nDone!")

    optimizer.param_groups[0]['lr'] = tmp_lr

    return losses


def trajectory_train_step(dataloader, model, pred_times, step_size, loss_fn, 
                                    optimizer, rk_method=rk4, print_stat=True):
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    model.train() #Tell model to run in train mode.
    
    for batch, (x, y) in enumerate(dataloader):
        
        pred = integrate_model_training(model, x, step_size, pred_times, rk_method=rk_method)
        
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad() # Reset the stored intermediate gradients
        loss.backward()
        optimizer.step()
        
        if 5 * batch % num_batches == 0 and print_stat:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
    return 


def trajectory_test(dataloader, model, pred_times, step_size, loss_fn, rk_method=rk4,
                                                                ln="\n", print_loss=True):
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    
    model.eval()
    
    test_loss = 0.0

    for x, y in dataloader:

        pred = integrate_model_training(model, x, step_size, pred_times, rk_method=rk_method)
        
        test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    
    if print_loss:
        print(f"Avg loss: {test_loss:>8f}", end=ln)
    
    return test_loss
    

def trajectory_train_loop(train_dataloader, test_dataloader, model, loss_fn, optimizer, 
                                            epochs, step_size=None, rk_method=rk4, lr=None):
                                                        
    pred_times = train_dataloader.dataset.dataset.pred_times

    if step_size is None:
        try:
            step_size = model.preferred_step_size
        except:
            raise ValueError("Integrator step size should be given as argument or available " + \
                             "in model.preferred_step_size.")

    losses = []

    tmp_lr = optimizer.param_groups[0]['lr']
    if lr is not None:
        optimizer.param_groups[0]['lr'] = lr

    for t in range(1,epochs+1):
        trajectory_train_step(train_dataloader, model, pred_times, step_size, loss_fn, 
                                        optimizer, rk_method=rk_method, print_stat=False)

        loss = trajectory_test(test_dataloader, model, pred_times, step_size, loss_fn, 
                                            rk_method=rk_method, ln="\n", print_loss=False)
        losses.append(loss)

        if 100*t % (10*epochs) == 0:
            print(f"Epoch {t:>3d} / {epochs} --------- ", end="")
            print(f"Avg loss: {loss:>8f}")
            
            
    print("\nDone!")

    optimizer.param_groups[0]['lr'] = tmp_lr

    return losses

