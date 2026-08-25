# Analyzer de Roupas — Visão Computacional com DeiT/ViT

Fine-tuning de um Vision Transformer (DeiT-Tiny) para classificação de imagens de peças de roupa, usando o dataset Fashion-MNIST.

## 📋 Sobre o projeto

Este projeto explora e adapta uma arquitetura **Vision Transformer (ViT)** — especificamente o **DeiT-Tiny (Data-efficient Image Transformer)** — para reconhecer 10 categorias de roupas a partir de imagens.

O trabalho parte da exploração de um modelo pré-treinado no ImageNet (inferência zero-shot) e evolui para um processo de **fine-tuning** completo, adaptando o modelo para uma tarefa de classificação específica.

## 🎯 Objetivos

1. Explorar um Vision Transformer pré-treinado (`timm`) com inferência zero-shot no ImageNet
2. Realizar fine-tuning do DeiT-Tiny no dataset Fashion-MNIST
3. Comparar o desempenho da arquitetura Transformer com abordagens de CNN (baseline)
4. ~~Salvar artefato para deploy via FastAPI~~ *(não realizado nesta etapa)*

## 🗂️ Dataset

**Fashion-MNIST** — 70.000 imagens em escala de cinza (28×28 pixels), divididas em 60.000 para treino e 10.000 para teste, distribuídas em 10 categorias:

| Classe | Categoria |
|---|---|
| 0 | Camiseta |
| 1 | Calça |
| 2 | Suéter |
| 3 | Vestido |
| 4 | Casaco |
| 5 | Sandália |
| 6 | Camisa |
| 7 | Tênis |
| 8 | Bolsa |
| 9 | Bota |

Carregado via `torchvision.datasets.FashionMNIST`.

## 🧠 Modelo

- **Arquitetura:** DeiT-Tiny (`deit_tiny_patch16_224`)
- **Parâmetros:** ~5,7 milhões
- **Origem dos pesos:** pré-treinado no ImageNet, via [`timm`](https://github.com/huggingface/pytorch-image-models)
- **Adaptação:** camada de classificação final trocada para 10 classes (`num_classes=10`)

Como o Fashion-MNIST é uma imagem em escala de cinza (1 canal) e o DeiT espera entrada RGB (3 canais), as imagens passam por uma conversão (`Grayscale(num_output_channels=3)`) antes do pré-processamento padrão do modelo.

## 🛠️ Tecnologias e bibliotecas

- **PyTorch** — treino e operações de tensor
- **timm** (PyTorch Image Models) — carregamento do modelo pré-treinado e configuração de pré-processamento
- **torchvision** — dataset Fashion-MNIST e transformações de imagem
- **transformers** (Hugging Face) — geração de artefato de configuração padronizado
- **Matplotlib** — visualização de métricas e previsões

## 🚀 Como rodar

1. Abra o notebook no [Google Colab](https://colab.research.google.com/)
2. Ative a GPU: **Ambiente de execução → Alterar tipo de ambiente de execução → GPU (T4)**
3. Execute as células em ordem, de cima para baixo (ou use **Ambiente de execução → Executar tudo**)

> ⚠️ Sem GPU, o treino roda em CPU e pode levar vários minutos por época. Confirme sempre se o print `Usando: cuda` aparece antes de iniciar o treino.

## 📊 Estrutura do notebook

| Seção | Conteúdo |
|---|---|
| Parte 1 | Exploração do DeiT-Tiny pré-treinado — listagem de modelos, inferência zero-shot no ImageNet |
| Parte 2 | Fine-tuning do DeiT-Tiny no Fashion-MNIST — dataset, loop de treino manual, métricas |
| Parte 3 | Visualização dos resultados — curvas de acurácia, previsões com nível de confiança |

## 📈 Resultados

- **Épocas de treino:** 5
- **Acurácia de treino:** ~96%
- **Melhor acurácia de validação:** ~93,8%

As curvas de treino e validação seguem próximas, sem sinais fortes de overfitting no número de épocas testado.

## 🔜 Próximos passos

- [ ] Empacotar o modelo treinado como artefato de deploy
- [ ] Criar uma API com **FastAPI** para servir previsões
- [ ] Gerar matriz de confusão para análise detalhada de erros por classe
- [ ] Testar hiperparâmetros adicionais (learning rate, épocas, data augmentation) para melhorar a acurácia

## 📄 Licença

Projeto acadêmico desenvolvido como parte da disciplina de Visão Computacional (FIAP).
