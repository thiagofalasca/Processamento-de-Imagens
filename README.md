
# Eficácia de Descritores: Hu Moments e LBP

Este projeto em Python emprega visão computacional para realizar a classificação de imagens em dois grupos, utilizando um modelo supervisionado. O foco central do programa é a extração de características por meio dos descritores Hu Moments e LBP. Inicialmente, o programa realiza a extração destas características, seguido pelo treinamento do modelo. Após o treinamento, o programa conduz testes utilizando os classificadores MLP (Multilayer Perceptron), Random Forest (RF) e SVM (Support Vector Machine). Ao finalizar os testes, é gerado um relatório contendo uma matriz de confusão, fornecendo uma análise detalhada da acurácia alcançada na classificação das imagens.ㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤ

## Autores

- Gustavo Aleixo
- Thiago Falasca
- João Victor Lobboㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤ


## Repositório

[![github](https://img.shields.io/badge/Repositório_Github-7?style=for-the-badge&logo=github&logoColor=whitek&color=black)](https://github.com/thiagofalasca/Processamento-de-Imagens)ㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤ

## Classificador e Acurácia

O processo de classificação neste projeto é conduzido por um modelo supervisionado utilizando os classificadores MLP (Multi-Layer Perceptron), Support Vector Machine (SVM), e Random Forest (RF). O resultado da classificação é apresentado em uma matriz de confusão.ㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤ



 **RESULTADOS OBTIDOS** 
| Hu Moments    ㅤ             | LBP                      |
| ---------------------------- | ------------------------ |
| MLP = 50.00% Acurácia   ㅤ   |  MLP = 89.29% Acuráciaㅤ |
| SVM = 53.57% Acurácia    ㅤ  |  SVM = 73.21% Acurácia ㅤ|
| RF = 60.71% Acurácia    ㅤ   |  RF = 98.21% Acuráciaㅤ  |

ㅤㅤ


## Instalação

Em um ambiente Linux, abra o terminal e execute os seguintes comandos:


```bash
# Instale o Python

sudo apt install python3
``` 



```bash
# Instale o gerenciador de pacotes do Python (pip)

sudo apt install python3
```

```bash
# Instale as bibliotecas necessárias

pip install split-folders
pip install opencv-python
pip install numpy
pip install scikit-image
pip install scikit-learn
pip install progress
pip install matplotlib
```





  
  
         
   
    
        
      