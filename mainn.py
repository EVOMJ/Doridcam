import cv2
import os
import urllib.request
import tempfile
import time
import numpy as np

# URL do Haar Cascade
URL_FACE = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

def baixar_cascade(nome_arquivo, url):
    if not os.path.exists(nome_arquivo):
        print(f"[INFO] Baixando {nome_arquivo}...")
        urllib.request.urlretrieve(url, nome_arquivo)

baixar_cascade("face.xml", URL_FACE)
face_cascade = cv2.CascadeClassifier("face.xml")

# Configurações
BASE_FACES = "faces"           # pasta com subpastas por pessoa
MIN_MATCHES = 25               # limiar de good matches para considerar como mesma pessoa (ajuste)
ORB_FEATURES = 2000            # mais features -> melhor chance de matching
DEBUG_SAVE = False             # se True salva imagem debug com keypoints em /tmp

# Inicializa ORB
orb = cv2.ORB_create(nfeatures=ORB_FEATURES)

# Pré-processamento do rosto (resize + CLAHE)
def preprocess_face(img_gray, size=(200,200)):
    try:
        face = cv2.resize(img_gray, size)
    except:
        face = img_gray.copy()
        face = cv2.resize(face, size, interpolation=cv2.INTER_LINEAR)
    # equalização adaptativa
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    face = clahe.apply(face)
    return face

# Carrega banco: retorna dict { nome: [descritores_img1, descritores_img2, ...] }
def carregar_banco():
    banco = {}
    if not os.path.exists(BASE_FACES):
        os.makedirs(BASE_FACES)
    for pessoa in os.listdir(BASE_FACES):
        pasta = os.path.join(BASE_FACES, pessoa)
        if not os.path.isdir(pasta):
            continue
        descritores_lista = []
        for arq in os.listdir(pasta):
            if arq.lower().endswith((".jpg", ".jpeg", ".png")):
                caminho = os.path.join(pasta, arq)
                img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = preprocess_face(img)
                kp, desc = orb.detectAndCompute(img, None)
                if desc is not None:
                    descritores_lista.append(desc)
        if len(descritores_lista) > 0:
            banco[pessoa] = descritores_lista
    print(f"[INFO] Banco carregado: {len(banco)} pessoas.")
    for k,v in banco.items():
        print(f"  - {k}: {len(v)} imagens")
    return banco

# compara um descritor do rosto capturado com a lista de descritores de uma pessoa
def comparar_com_pessoa(desc_rosto, lista_descritores):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    melhor = 0
    for desc_banco in lista_descritores:
        if desc_banco is None or desc_rosto is None:
            continue
        # matches KNN para aplicar ratio test
        try:
            matches = bf.knnMatch(desc_rosto, desc_banco, k=2)
        except:
            continue
        good = 0
        for m_n in matches:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:  # ratio test de Lowe
                good += 1
        if good > melhor:
            melhor = good
    return melhor

# salva debug (opcional)
def salvar_debug_kp(img_gray, kp, caminho):
    if not DEBUG_SAVE:
        return
    img_kp = cv2.drawKeypoints(img_gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    try:
        tmp = os.path.join(tempfile.gettempdir(), caminho)
        cv2.imwrite(tmp, img_kp)
        print(f"[DEBUG] Salvo: {tmp}")
    except Exception as e:
        print("[DEBUG] Erro ao salvar debug:", e)

# função principal
def capturar_rosto(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Erro ao abrir a câmera!")
        return

    print("Câmera aberta! Pressione ESPAÇO para capturar | Q para sair.")
    banco = carregar_banco()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60,60))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        cv2.imshow("Detector de Rosto", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == 32:  # ESPAÇO
            if len(faces) == 0:
                print("Nenhum rosto detectado!")
                continue

            # uso a primeira face detectada
            x, y, w, h = faces[0]
            rosto_gray = gray[y:y+h, x:x+w]
            rosto_proc = preprocess_face(rosto_gray)

            kp, desc_rosto = orb.detectAndCompute(rosto_proc, None)
            salvar_debug_kp(rosto_proc, kp, "debug_rosto.jpg")

            if desc_rosto is None:
                print("Não foi possível extrair descritores do rosto (desc_rosto is None).")
                continue

            # tenta identificar comparando com cada pessoa no banco
            melhor_nome = None
            melhor_score = 0

            for nome, lista_desc in banco.items():
                score = comparar_com_pessoa(desc_rosto, lista_desc)
                print(f"[DEBUG] Melhor match para {nome} = {score}")
                if score > melhor_score:
                    melhor_score = score
                    melhor_nome = nome

            print(f"[INFO] Melhor nome: {melhor_nome} | Melhor score: {melhor_score}")

            # decisão: se passar do limiar, considera identificado
            if melhor_score >= MIN_MATCHES:
                print(f"\nROSTO IDENTIFICADO: {melhor_nome} (matches: {melhor_score})")
                autorizacao = input("Autorizar? (S/N): ").strip().upper()
                if autorizacao == "S":
                    print("ACESSO AUTORIZADO!\n")
                else:
                    print("ACESSO NEGADO!\n")
            else:
                print("\nRosto NÃO reconhecido.")
                salvar = input("Deseja salvar essa face para cadastro? (S/N): ").strip().upper()
                if salvar == "S":
                    nome_novo = input("Digite o nome da pessoa (sem espaços especiais): ").strip()
                    if nome_novo == "":
                        nome_novo = f"desconhecido_{int(time.time())}"
                    pasta = os.path.join(BASE_FACES, nome_novo)
                    if not os.path.exists(pasta):
                        os.makedirs(pasta)
                    caminho = os.path.join(pasta, f"img_{int(time.time())}.jpg")
                    cv2.imwrite(caminho, rosto_proc)
                    print(f"[INFO] Salvo em: {caminho}")
                    # recarrega banco
                    banco = carregar_banco()
                else:
                    print("Não salvo.")

    cap.release()
    cv2.destroyAllWindows()

# ajuste da sua URL
url_camera = "http://172.25.252.61:4747/video"
capturar_rosto(url_camera)
