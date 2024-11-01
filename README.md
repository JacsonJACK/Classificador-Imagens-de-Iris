# Classificador-Imagens-de-Iris
 
Classificação de Imagens de Íris

Este projeto utiliza algoritmos de visão computacional para classificar imagens de íris em duas categorias: saudável e com problema. O código realiza a extração de características das imagens utilizando LBP (Local Binary Patterns) e HOG (Histogram of Oriented Gradients) e treina um modelo de SVM para realizar a classificação.

Funcionalidades
Extração de Características: Usa LBP e HOG para capturar informações visuais das imagens.
Treinamento do Modelo: Um modelo de SVM com kernel radial é treinado para diferenciar imagens saudáveis das com problemas.
Classificação e Visualização: As imagens são classificadas e as que apresentarem problemas são exibidas.

Estrutura do Projeto
As imagens devem estar organizadas nas seguintes pastas:
Iris_problema: Contém as imagens de íris com problemas.
Iris_saudavel: Contém as imagens de íris saudáveis.
Iris_diversas: Contém imagens de íris que serão classificadas pelo modelo treinado.

Pré-requisitos
Para executar o código, instale as seguintes bibliotecas:
pip install opencv-python numpy matplotlib scikit-image scikit-learn

Coloque as imagens nas pastas apropriadas conforme a estrutura do projeto.
O código carregará as imagens das pastas Iris_problema e Iris_saudavel, extrairá as características, treinará o modelo e realizará a classificação nas imagens em Iris_diversas.

Saída do Programa
A acurácia do modelo treinado será exibida no console.
Imagens classificadas como "com problema" em Iris_diversas serão mostradas com o caminho do arquivo indicado.
No console, será listada a localização de cada imagem classificada como com problema.

Exemplo de Saída
Acurácia no conjunto de teste: 0.85
Imagens com problema na pasta Iris_diversas:
C:/Users/Iris_diversas/imagem1.jpg
C:/Users/Iris_diversas/imagem2.jpg

Estrutura do Código
O código é dividido nas seguintes etapas:
Carregamento e Pré-processamento das Imagens: Carrega imagens de íris, redimensiona e equaliza o histograma.
Extração de Características: Calcula histogramas LBP e características HOG.
Treinamento do Modelo: Um modelo SVM é treinado usando um conjunto de treino e teste.
Classificação de Novas Imagens: Classifica as imagens na pasta Iris_diversas e exibe aquelas identificadas como contendo problemas.
