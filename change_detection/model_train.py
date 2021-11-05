
# model training

def model_train(dataload,model,loss_fn,optimizer):
    size = dataload.dataset
    for batch ,(T1,T2,label) in enumerate(dataload):
        pred = model()
        loss = loss_fn(pred,y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    pass