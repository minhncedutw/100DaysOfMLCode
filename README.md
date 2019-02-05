# 100DaysOfMLCode

## Day 1(03March2019): Study how pytorch do classification? log_softmax?
 - Q: How to do fastai PointNet?
 - Q: Why pointnet_pytorch they do not directly use cross-entropy? Is it related to special shape of data? 
 Or is it still able to use cross-entropy?
 - A: We can directly define cross-entropy, need not use `log_softmax` and `nll_loss`. Just call 
 `loss = F.cross_entropy(outputs, target)`
 - Q: cross-entropy = log_softmax + nll_loss?
 - A: YES. CrossEntropyLoss() is combination combines LogSoftmax() and NLLLoss() 
 [source](https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss) 
 - Q: How to do softmax? cross-entropy? input/output shape of each one?
 - A: cross-entropy and log_softmax has the same input shape: (N, C, d1, d2, .. , dK)
 - Q: Can it do softmax without .view(...)?

- People can directly define cross-entropy. After that, they can calculate loss by calling 
`loss = F.cross_entropy(outputs, target)`. So they need not use log_softmax and nll_loss.
- Why pointnet_pytorch they do not directly use cross-entropy? 

## Day 2(04March2019): Code PointNet using cross-entropy instead of log_softmax on pytorch framework

## Day 3(05March2019): Code PointNet on fastai framework

## Questions: 

 - How to run pytorch network on cuda?