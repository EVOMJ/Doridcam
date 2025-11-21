import cv2
import os
import urllib.request
import tempfile
import time
import numpy as np

# URLs dos Haar Cascades
URL_FACE = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

def baixar_cascade(nome_arquivo, url):
    if not os.path.exists(nome_arquivo):
        print(f"[INFO] Baixando {nome_arquivo}...")
        urllib.request.urlretrieve(url, nome_arquivo)

baixar_cascade("face.xml", URL_FACE)

face_cascade = cv2.CascadeClassifier("face.xml")


# ---------------------------
# FUNÇÃO: CARREGAR BANCO DE FACES
# ---------------------------
def carregar_banco():
    banco = {}
    for arq in os.listdir():
        if arq.lower().endswith(".jpg"):
            img = cv2.imread(arq, 0)
            if img is not None:
                banco[arq.replace(".jpg", "")] = img
    return banco


# ---------------------------
# FUNÇÃO: COMPARAR ROSTOS
# ---------------------------
def comparar_rosto(img1, img2):
    img1 = cv2.resize(img1, (200, 200))
    img2 = cv2.resize(img2, (200, 200))

    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return score


# ---------------------------
# FUNÇÃO PRINCIPAL
# ---------------------------
def capturar_rosto(url):
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("Erro ao abrir a câmera!")
        return

    print("Câmera aberta! Pressione ESPAÇO para capturar.")

    banco = carregar_banco()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Detector de Rosto", frame)

        key = cv2.waitKey(1) & 0xFF

        # ---------------------------
        # CAPTURAR FOTO
        # ---------------------------
        if key == 32:  # ESPAÇO
            print("\nIniciando contagem...")
            for i in range(3, 0, -1):
                print(f"Capturando em {i}...")
                time.sleep(1)

            # Recorta apenas o rosto
            if len(faces) == 0:
                print("Nenhum rosto detectado!")
                continue

            x, y, w, h = faces[0]
            rosto = gray[y:y+h, x:x+w]

            # ---------------------------
            # IDENTIFICAÇÃO
            # ---------------------------
            melhor_nome = None
            melhor_score = 0

            for nome, img_banco in banco.items():
                score = comparar_rosto(rosto, img_banco)
                if score > melhor_score:
                    melhor_score = score
                    melhor_nome = nome

            if melhor_score > 0.85:  # limite de similaridade
                print(f"\nROSTO IDENTIFICADO como: **{melhor_nome}** (similaridade: {melhor_score:.2f})")

                autorizacao = input("Autorizar? (S/N): ").strip().upper()

                if autorizacao != "S":
                    print("Acesso negado.")
                    continue

                print("Acesso autorizado!\n")

            else:
                print("\nRosto NÃO reconhecido.")

            # ---------------------------
            # SALVAR FOTO
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, "foto_temp.jpg")

            cv2.imwrite(temp_path, rosto)

            print("\nDigite o nome da pessoa:")
            nome = input("Nome: ").strip()

            if nome == "":
                nome = "foto_sem_nome"

            nome_arquivo = f"{nome}.jpg"
            os.replace(temp_path, nome_arquivo)

            print(f"Foto salva como: {nome_arquivo}\n")

            banco = carregar_banco()  # recarrega banco após novo cadastro

        # sair
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



# SUA URL DA CAMERA
url_camera = "http://172.25.252.61:4747/video"
capturar_rosto(url_camera)
