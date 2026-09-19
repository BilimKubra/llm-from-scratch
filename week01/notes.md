# Hafta 1 - Temel Kavramlar

## Hedef
Neuron, katman, parametre, loss fonksiyonu, forward pass, gradient descent

## Kaynak
Karpathy - Neural Networks: Zero to Hero (ilk video)

## Deney Notları

## Sonuç

## Kod Mantığım (kendi notlarım)
- Neuron: her girdiyi bir agirlikla carpip topluyorum (agirlikli toplam = z)
- z = (x1*w1 + x2*w2 + x3*w3) + bias
- zip() ile iki listeyi (inputs, weights) birebir eslestirdim
- bias, tum girdiler 0 olsa bile neuronun bir ciktisi olmasini sagliyor
- sigmoid ile z'yi 0-1 arasina sikistirdim (olasilik gibi yorumlanabilsin diye)
- sigmoid(z) = 1 / (1 + e^(-z))

## Deney Sonucu
- Girdi: [1.0, 2.0, 3.0], Agirlik: [0.2, -0.5, 0.1], Bias: 0.7
- z ve sigmoid ciktisini terminalde gordum, degerleri buraya not al
