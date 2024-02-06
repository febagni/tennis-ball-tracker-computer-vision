# Saudações Patrióticas

Segue anexo minhas anotações pro bagni ver hoje de manhã: 


## Problemas do Bounce = 1
Temos: 233 frames, tendo então 232 entradas no dataframe de bounce (não tem como saber a "velocidade" nos extremos). 

Após isso, o nosso homi cria 60 features: pra cada frame, ele cria: 20 colunas que tem os 20 valores da coordenada X antes do frame. 20 valores com as ultimas 20 coordenadas de Y e o mesmo pras velocidades. 

Depois disso, ele faz uma transformação bizarra que possibilita usar uma biblioteca bizarra q eu nunca vi na vida de ML pra series temporais, da pra fazer a mesma coisa q ele fez mas com Machine Learning normal, acho q eh legal tentar. 

O nosso problema: 
* aparentemente o nosso X está normal e não tem valores None, mas por algum motivo na hora de colocar no pipeline que o cara tem feito, o negócio buga, eu acho q se trata de algum problema do próprio código do cara, so que eu to procurando no github dele e ele n deixou o q ele adicionou nesse pipeline, mas eu tento abrir ele ou ver os parametros dele e ele n funciona. Acho q eh problema do cara querer misturar a biblioteca nova ali de time series com o sklearn pipeline. 


Na minha conclusão, n tem mt o q fazer pra falar a real, eu acho q em alguma parte do pipeline q ele criou, ele cria valores None e o classificador n funciona com NaN.

Talvez o negócio do cara real funcionava, mas por conta de atualização do sklearn ou do sktime o negócio tenha parado de funcionar msm

## Conclusoes

* Shadow o ouriço é um filho da puta do caralho
* Isso eh uma coisa boa tendo em vistq q a ideia do nosso projeto eh realmente implementar isso e agora podemos fazer isso nos mesmos cagando pra o q o cara inventou ali (eru acho real q foi um semi surto do nosso mano)




