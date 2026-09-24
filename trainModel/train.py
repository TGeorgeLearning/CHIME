class Train:
  def __init__(self,max_epochs,gpuOn=False):
    self.me = max_epochs
    self.gpuAct = gpuOn
    self.finaloa=0
    self.waiting=0
      
  def fit(self,model,trainData):
    self.model=model
    self.train_dataloader = trainData

    self.optim = model.config_optimiser()

    self.epoch=0

    for self.epoch in range(self.me):
      print(f"\n--- Epoch {self.epoch+1}/{self.me} ---") #
      self.fit_epoch()
      if (self.waiting==7):
          print("Waiting reached 7 epochs, ending training")
          break


  def fit_epoch(self):
    totalLoss=0
    count=0
    model.train() 
    model.currEpoch+=1
    accu=torch.zeros(28, device=gpu()) 
    for batch in self.train_dataloader:
      if (self.gpuAct):
        batch = [item.to(gpu()) for item in batch]
      loss,acc = self.model.trainStep(batch)
      totalLoss+=loss.detach()
      accu+=acc
      count+=1
      self.optim.zero_grad()
      with torch.no_grad():
        loss.backward()
        self.optim.step()

    self.trainArr.append(totalLoss/count)
    print("Loss of", totalLoss/count)
    print("Train Accuracy array")
    print((accu / count).reshape(7, 4))
    loss=0
    accu=torch.zeros(28, device=gpu()) 
    count=0
    model.eval()
    with torch.no_grad():
        calcoa = self.model.computeValAccuracy()
        if (calcoa>self.finaloa):
          self.finaloa=calcoa
          self.waiting=0
          torch.save(model.state_dict(), '') # Make sure to define where you want the model checkpoints to be stored
          print(f"Saved the best model checkpoint, reached OA: {calcoa}")
        else:
            print(f"Best oa {self.finaloa}, achieved {calcoa}")
            self.waiting+=1
        
        