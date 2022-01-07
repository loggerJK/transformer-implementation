

# Pytorch에서 Layer를 확인하는 법

Xavier Uniform으로 layer들을 초기화해야해서, 모델을 선언 후 layer.weight들을 불러와서 초기화해주려는 방법을 쓰려고 했다. 그런데 특정 layer들이 제대로 불러와지지 않는 문제를 확인했다.

```python
for param in encoder_layer.named_parameters():
    print(param[0])
```
```
multiheadattention.fcQ.weight
multiheadattention.fcQ.bias
multiheadattention.fcK.weight
multiheadattention.fcK.bias
multiheadattention.fcV.weight
multiheadattention.fcV.bias
multiheadattention.fcOut.weight
multiheadattention.fcOut.bias
ffn.fc1.weight
ffn.fc1.bias
ffn.fc2.weight
ffn.fc2.bias
layerNorm.weight
layerNorm.bias
```

 본 코드의 Decoder Architecture 부분을 확인해보면 다음과 같은 부분이 있다.

```python
self.dec_layers = []
    for i in range(N):
        layer = DecoderLayer(hidden_dim, num_head, inner_dim)
        self.dec_layers.append(layer)
```

이러한 식으로 리스트 안에 layer를 선언한 경우에는, 제대로 layer가 불러와지지 않는다.

```python
for layer in model.decoder.named_parameters():
    print(layer[0])
```
```
embedding.weight
finalFc.weight
finalFc.bias
```

다음과 같이 ``model.decoder`` 객체 안의 ``dec_layers``으로 직접 접근해서 ``named_parameters()``를 콜해야만 레이어들이 불러와지는 것을 확인할 수 있다. ``named_parameters()``가 아닌 ``children()`` 메소드를 이용한다고 하더라도 결과는 동일하다. (불러와지지 않는다)

# Pytorch Layer Initialization
## 첫번째 : apply() 함수 이용하기

```python
def apply_xavier(layer):
    if hasattr(layer, 'weight'):
        print(layer)
        torch.nn.init.xavier_uniform_(layer.weight)

encoder_layer=EncoderLayer(128,8,2048)
encoder_layer.apply(apply_xavier)
```
```
Linear(in_features=128, out_features=128, bias=True)
Linear(in_features=128, out_features=128, bias=True)
Linear(in_features=128, out_features=128, bias=True)
Linear(in_features=128, out_features=128, bias=True)
Linear(in_features=128, out_features=2048, bias=True)
Linear(in_features=2048, out_features=128, bias=True)
LayerNorm((128,), eps=1e-05, elementwise_affine=True)
```

## 두번째 : named_parameters() 함수 이용하기

named_parameters() 함수는 (param_name, param_weight) 형태의 튜플을 반환한다. 

```python
for param in encoder_layer.named_parameters():
  print(param[0])
```
```
multiheadattention.fcQ.weight
multiheadattention.fcQ.bias
multiheadattention.fcK.weight
multiheadattention.fcK.bias
multiheadattention.fcV.weight
multiheadattention.fcV.bias
multiheadattention.fcOut.weight
multiheadattention.fcOut.bias
ffn.fc1.weight
ffn.fc1.bias
ffn.fc2.weight
ffn.fc2.bias
layerNorm.weight
layerNorm.bias
```

Xavier Uniform Initilization을 이용하고자 한다면 다음과 같이 조건식을 추가하여 초기화하면 된다. (bias와 nn.layerNorm()은 초기화 대상이 아니므로 제외해준 모습을 볼 수 있다.)

```python
for layer in encoder_layer.named_parameters():
    if 'weight' in layer[0] and 'layerNorm' not in layer[0]:
        print(layer[0])
        torch.nn.init.xavier_uniform_(layer[1])
```
```
multiheadattention.fcQ.weight
multiheadattention.fcK.weight
multiheadattention.fcV.weight
multiheadattention.fcOut.weight
ffn.fc1.weight
ffn.fc2.weight
```

# Pytorch에서 List 형식으로 layer 선언하기
처음에 작성했던 코드는 다음과 같다.
```python
class Decoder(nn.Module):
    def __init__ (self, N, hidden_dim, num_head, inner_dim):
        super().__init__()

        self.dec_layers = []
        for i in range(N):
            self.dec_layers.append(DecoderLayer(hidden_dim, num_head, inner_dim))

```
이 코드는 문제가 있는 코드이다. 왜일까? 저렇게 단순히 Python 리스트에 레이어를 집어넣어서 사용하면 Pytorch가 layer를 제대로 인식하지 못하는 상황이 벌어지기 때문이다. 이 말인 즉, 상위 layer에서 ``children()``을 호출해도 저 ``self.dec_layers``안의 layer들은 호출되지 않는다는 이야기이다. 당연히 ``model.parameters()``를 호출해도 저 layer들의 parameter들은 누락되게 되고, 학습을 해도 optimizer가 최적화하지 않는 치명적인(!) 상태가 된다. (당연하다, optimizer에 parameter가 등록되어 있지 않으니까.)

그러면 어떻게 해야할까?

## nn.ModuleList
이를 위해서, Python은 ``nn.ModueList``를 제공한다. 사용법은 다음과 같다

```python
self.dec_layers = nn.ModuleList([DecoderLayer(hidden_dim, num_head, inner_dim) for _ in range(N)])
```

그러면 Pytorch에서 정상적으로 layer들을 인식한다. 쓸 때는 일반적인 반복문처럼 ```for layer in self.dec_layers:``` 같이 사용하면 된다.


# Pytorch에서 += 연산자의 위험성

[참고 사이트](https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/4)

Pytorch에서 layer를 짤 때 다음과 같은 코드는 조심해야 한다.

```python
    def forward(self, input, mask = None):

        # input : (bs, seq_len, hidden_dim)
        
        # encoder attention
        # uses only padding mask
        output = self.multiheadattention(srcQ= input, srcK = input, srcV = input, mask = mask)
        output = self.dropout1(output)
        output += input
        output = self.layerNorm(output)

        output_ = self.ffn(output)
        output_ = self.dropout2(output_)
        output += output
        output = self.layerNorm(output)

        # output : (bs, seq_len, hidden_dim)
        return output
```

왜냐? 이 ``+=`` 연산자가 바로 inplace 연산자이기 때문이다. 따라서 이를 이용해서 layer를 짜고 ``loss.backward()``를 하면 Pytorch가 ``One of the variables needed for gradient computation has been modified by an inplace operation`` 에러를 내뿜게 된다. 디버깅하기 힘드니까 조심하자.... Pytorch를 사용하면서 느끼는 건 잘 모르면 진짜 그냥 안전하게 짜는게 에러 안나고 베스트라는 것이다. 코드 길이 줄이겠다고 ``+=`` 썼다가 에러 잡느라 몇시간을 날렸다....