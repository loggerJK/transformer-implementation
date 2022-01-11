# transformer-implementation

[Vanilla Transformer NLP](https://arxiv.org/abs/1706.03762) 모델을 구현합니다.
모델은 `Training.ipynb`에 구현되어 있습니다.

[블로그](https://loggerjk.github.io/pytorch/Transformer/)


-   Framework : Pytorch
-   Task : EN -> KO Translation
-   Dataset : [korean-parallel-corpora/bible](https://github.com/jungyeul)

    -   Sentencepice
    -   Vocab Size : 10K
    -   Train : Valid = 9 : 1

-   Training
    -   Encoder/Decoder : 2
    -   hidden_dim = 256
    -   inner_dim = 512
    -   Epoch : 70
    -   Learning Rate : 1e-4
    -   Scheduler : CosineAnnealingLR (Tmax = 100, min = 1e-5)


<p align="center">
<img src=https://i.imgur.com/CFMuitM.png alt=img width=80% height=80% />
</p>


-   Training Result
    -   Train_Loss : 2.64
    -   Train accuracy : 0.203
    -   Valid_Loss : 4.46
    -   Valid accuracy : 0.136

-   Good Example

```
en =  " 'This is what the Sovereign LORD says: In the first month on the first day you are to take a young bull without defect and purify the sanctuary.
answer =  "나 주 하나님이 말한다. 너는 첫째 달 초하루에는 언제나 소 떼 가운데서 흠 없는 수송아지 한 마리를 골라다가 성소를 정결하게 하여라.
ko = ['나 주 하나님이 말한다. 그 날에는 수송아지 일곱 마리와 숫양 두 마리와 일 년 된 어린 숫양 한 마리를 흠 없는 것으로 바쳐라.']

en =  Solomon reigned in Jerusalem over all Israel forty years.
answer =  솔로몬은 예루살렘에서 사십 년 동안 온 이스라엘을 다스렸다.
ko = ['솔로몬은 예루살렘에서 마흔 해 동안 다스렸다.']

en =  then hear from heaven their prayer and their plea, and uphold their cause.
answer =  주께서는 하늘에서 그들의 기도와 간구를 들으시고, 그들의 사정을 살펴보아 주십시오.
ko = ['그러나 주님은, 하늘에서 그들의 기도와 간구를 들으시고, 그들의 사정을 살펴 주십시오.']
```

-   Bad Example

```
en =  Obed-Edom also had sons: Shemaiah the firstborn, Jehozabad the second, Joah the third, Sacar the fourth, Nethanel the fifth,
answer =  오벳에돔의 아들은, 맏아들 스마야와, 둘째 여호사밧과, 셋째 요아와, 넷째 사갈과, 다섯째 느다넬과,
ko = ['오벳에돔과 아사의 아들 여호하난이 보수하였는데, 그 다음은 단에서부터 스바와 드라빔과 스바와 드라빔과 스바와 드라빔과 스바와 드라빔과 스바와 드라빔과 스바와 드라빔과 스바와 드라빔과 스바와 드라빔과 스바와 드라빔과 스바와 드라빔과 스바와 드라빔이다.']

en =  "Go down, sit in the dust, Virgin Daughter of Babylon; sit on the ground without a throne, Daughter of the Babylonians. No more will you be called tender or delicate.
answer =  처녀 딸 바빌론아, 내려와서 티끌에 앉아라. 딸 바빌로니아야, 보좌를 잃었으니, 땅에 주저앉아라. 너의 몸매가 유연하고 맵시가 있다고들 하였지만, 이제는 아무도 그런 말을 하지 않을 것이다.
ko = ['"너는 바빌론 도성 바빌론 도성아, 바빌론 도성아, 바빌론 도성 안에 있는 도성 안에 있는 네 오른손에는 칼이나 쳐라. 네 오른손에는 칼이나 기근이나 기근이나 기근이나 기근이나 기근이나 기근이나 기근이나 굶은 아니다.']
```
