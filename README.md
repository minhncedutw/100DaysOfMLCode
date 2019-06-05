# 100DaysOfTechLearning

#### --------------------------------------------------
## Day 1(2019June03): Watch [3 Limits of Artificial Intelligence](https://youtu.be/f5YvatzVWxA)

3 giới hạn của AI:
- thiếu nhân quả(lack causal reasoning)
- thiếu diễn giải(lack interpretability)
- bị tổn thương bởi các ví dụ đối nghịch(vulnerable to adversarial example)


#### --------------------------------------------------
## Day 2(2019June04): Learn "[Explainable AI]"(which solves AI problem [Lacking Interpretability]")

Article: [An introduction to explainable AI, and why we need it](https://www.freecodecamp.org/news/an-introduction-to-explainable-ai-and-why-we-need-it-a326417dd000/)

**Why the need for explainable models?**
- Sometimes: neural networks deliver the wrong prediction; the neural network may be fooled by adversarial examples.
- Thus, we want to know that is the model trustable, and when it is trustable.
- some domains prefer the reasons for predictions(such as banking, health care, ...) than prediction only.

**Approach**
- Reversed Time Attention Model (RETAIN): 2 RNNs combined together to do 2-in-1: aggregate information to predict; use attention mechanism on each information to understand what the NN was focusing on.
- Local Interpretable Model-Agnostic Explanations (LIME)(**fairly common**): perturbing the inputs and watching how doing so affects the model's output. Based on that to build up a picture of which inputs the model is focusing on to make predictions.
- Layer-wise Relevance Propagation (LRP): measure the relevance of each neural in a layer with the neutrals in previous layers and finally determine the relevance of individual inputs to output, then extract a meaningful subset of inputs that most affect the prediction.

#### --------------------------------------------------
## Day 3(2019June05): Practice "[xAI - LIME]"

**Sample code** [LIME](https://github.com/marcotcr/lime)

#### --------------------------------------------------
## Day 4(2019June06): Teach "[Lacking Interpretability - xAI]"


[Lacking Interpretability]: https://github.com/minhncedutw/100DaysOfMLCode/commit/6d5322baa27c68c1d595a876fb792dbdb5557fad
[Explainable AI]: https://www.freecodecamp.org/news/an-introduction-to-explainable-ai-and-why-we-need-it-a326417dd000/
