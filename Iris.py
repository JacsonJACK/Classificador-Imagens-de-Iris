import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Função para extrair características combinando LBP e HOG
def extract_features(path):
    img = cv2.imread(path, 0)  # Carregar em escala de cinza

    # Verificar se a imagem foi carregada corretamente
    if img is None:
        #print(f"Erro ao carregar a imagem: {path}")
        print("Erro ao carregar a imagem: %s" % path)
        return None

    # Redimensionar a imagem para 128x128 pixels
    img = cv2.resize(img, (128, 128))

    # Equalizar a imagem para melhorar o contraste
    img = cv2.equalizeHist(img)

    # Extrair LBP (calculando histograma)
    lbp = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()  # Certificando-se de que é um vetor 1D
    lbp_hist = lbp / np.sum(lbp)  # Normalizando o histograma

    # Extrair HOG
    hog_features = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False,
                       channel_axis=None)

    # Concatenar as características
    features = np.concatenate((lbp_hist, hog_features))
    return features


# Caminhos para as pastas
path_problema = r"C:/Users/Iris_problema"
path_saudavel = r"C:/Users/Iris_saudavel"
path_diversas = r"C:/Users/Iris_diversas"

# Listas para armazenar as características e os rótulos
X = []
y = []

# Carrega imagens de íris com problema e extrai características
for img_path in glob.glob(path_problema + "/*.jpg"):
    features = extract_features(img_path)
    if features is not None:
        X.append(features)
        y.append(1)

# Carrega imagens de íris saudáveis e extrai características
for img_path in glob.glob(path_saudavel + "/*.jpg"):
    features = extract_features(img_path)
    if features is not None:
        X.append(features)
        y.append(0)

# Verifique o tamanho de cada vetor de características
for i, features in enumerate(X):
    print("Tamanho das características da imagem %d: %s" % (i, features.shape))

# Convertendo listas para arrays NumPy
X = np.array(X)
y = np.array(y)

# Divide em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treina o modelo SVM com kernel radial
model = SVC(kernel='rbf', gamma='scale', random_state=42)
model.fit(X_train, y_train)

# Avalia o modelo no conjunto de teste
y_pred = model.predict(X_test)
print("Acurácia no conjunto de teste:", accuracy_score(y_test, y_pred))

# Classificação das imagens em Iris_diversas
imagens_com_problema = []

for img_path in glob.glob(path_diversas + "/*.jpg"):
    features = extract_features(img_path)
    if features is not None:
        prediction = model.predict([features])[0]
        if prediction == 1:
            imagens_com_problema.append(img_path)
            # Exibir a imagem com problema
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.title("Imagem com problema: %s" % img_path)
            plt.axis('off')
            plt.show()

# Lista as imagens com problema
print("Imagens com problema na pasta Iris_diversas:")
for img in imagens_com_problema:
    print(img)
