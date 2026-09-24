def init_weights(m):
     if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


model = Flower(lr=0.0005).to(gpu()) 

initializeLayers = torch.randn(1,6, 288, 172).to(gpu())
_ = model(initializeLayers)

model.apply(init_weights)

trainer = Train(300,gpuOn=True)
trainer.fit(model, trainLoader,None)