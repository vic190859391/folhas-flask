from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import webbrowser
import threading 
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter, median_filter

app = Flask(__name__, template_folder='templates', static_folder='static')

def analisar_folha(imagem, indice_selecionado=None):
    imagem_np = np.array(imagem)
    imagem_cv = cv2.cvtColor(imagem_np, cv2.COLOR_RGB2BGR)

    hsv = cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2HSV)
    fundo_claro = cv2.inRange(hsv, (0, 0, 210), (180, 40, 255))
    sombra_escura = cv2.inRange(hsv, (0, 0, 0), (180, 70, 70))
    mascara_fundo = cv2.bitwise_or(fundo_claro, sombra_escura)
    mascara_folha = cv2.bitwise_not(mascara_fundo)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara_folha = cv2.morphologyEx(mascara_folha, cv2.MORPH_OPEN, kernel)
    mascara_folha = cv2.morphologyEx(mascara_folha, cv2.MORPH_CLOSE, kernel)

    imagem_folha = cv2.bitwise_and(imagem_cv, imagem_cv, mask=mascara_folha)
    hsv_folha = cv2.cvtColor(imagem_folha, cv2.COLOR_BGR2HSV)

    mask_saudavel = cv2.inRange(hsv_folha, (35, 40, 40), (90, 255, 255))
    mask_doente = cv2.inRange(hsv_folha, (20, 50, 90), (35, 255, 255))
    mask_morta = cv2.inRange(hsv_folha, (0, 30, 30), (25, 255, 120))

    imagem_resultado = imagem_np.copy()
    imagem_resultado[mask_saudavel > 0] = [0, 255, 0]
    imagem_resultado[mask_doente > 0] = [255, 255, 0]
    imagem_resultado[mask_morta > 0] = [139, 69, 19]

    area_saudavel = int(np.sum(mask_saudavel > 0))
    area_doente = int(np.sum(mask_doente > 0))
    area_morta = int(np.sum(mask_morta > 0))
    area_total = area_saudavel + area_doente + area_morta

    imagem_float = imagem_np.astype('float32')
    R = imagem_float[:, :, 0]
    G = imagem_float[:, :, 1]
    B = imagem_float[:, :, 2]
    G_raw = G.copy()

    mascara_valida = (
        (mascara_folha > 0) &
        (G > R) & (G > B) &
        (G_raw > 30)
    )

    indices_vegetacao = {}
    imagem_indice_colorida = None
    fundo_branco = np.ones_like(imagem_np, dtype=np.uint8) * 255

    def normalizar_e_colorir(matriz, mascara, nome):
        valores = matriz[mascara]
        valores = valores[np.isfinite(valores)]
        valores = valores[(valores >= -1.0) & (valores <= 1.0)]
        valores = valores[np.abs(valores) > 0.02]

        if len(valores) == 0:
            valores = np.array([0.0])

        media = float(np.mean(valores))
        desvio = float(np.std(valores))
        cv_percent = float((desvio / media) * 100) if media != 0 else 0.0

        indices_vegetacao[nome] = {
    'min': float(np.min(valores)),
    'media': media,
    'max': float(np.max(valores)),
    'observacao': 'Valores estatísticos calculados apenas sobre a área da folha, com exclusão do fundo.'
}


        p1, p99 = np.percentile(valores, [1, 99])
        valores_clipped = np.clip(matriz, p1, p99)

        norm = np.zeros_like(matriz)
        norm[mascara] = ((valores_clipped[mascara] - p1) / (p99 - p1) * 255)
        norm_uint8 = norm.astype(np.uint8)

        cmap = cm.get_cmap('RdYlGn')
        color_rgb = (cmap(norm_uint8 / 255.0)[..., :3] * 255).astype(np.uint8)
        return np.where(mascara[..., None], color_rgb, fundo_branco)

    if indice_selecionado == 'vari':
        VARI = (G - R) / (G + R - B + 1e-6)
        VARI_suavizado = median_filter(gaussian_filter(VARI, sigma=2), size=3)
        imagem_indice_colorida = normalizar_e_colorir(VARI_suavizado, mascara_valida, 'vari')

    elif indice_selecionado == 'gli':
        GLI = (2 * G - R - B) / (2 * G + R + B + 1e-6)
        GLI_suavizado = median_filter(gaussian_filter(GLI, sigma=2), size=3)
        imagem_indice_colorida = normalizar_e_colorir(GLI_suavizado, mascara_valida, 'gli')

    elif indice_selecionado == 'ngrdi':
        NGRDI = (G - R) / (G + R + 1e-6)
        NGRDI_suavizado = median_filter(gaussian_filter(NGRDI, sigma=2), size=3)
        imagem_indice_colorida = normalizar_e_colorir(NGRDI_suavizado, mascara_valida, 'ngrdi')

    buffer = BytesIO()
    Image.fromarray(imagem_resultado).save(buffer, format="PNG")
    imagem_base64 = base64.b64encode(buffer.getvalue()).decode()

    imagem_indice_base64 = None
    if imagem_indice_colorida is not None:
        buffer_indice = BytesIO()
        Image.fromarray(imagem_indice_colorida).save(buffer_indice, format="PNG")
        imagem_indice_base64 = base64.b64encode(buffer_indice.getvalue()).decode()

    return {
        'imagem': imagem_base64,
        'imagem_indice': imagem_indice_base64,
        'area_total': area_total,
        'area_saudavel': area_saudavel,
        'area_doente': area_doente,
        'area_morta': area_morta,
        'indices_vegetacao': indices_vegetacao,
        'indice_nome': indice_selecionado
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analisar', methods=['POST'])
def analisar():
    if 'image' not in request.files:
        return jsonify({'erro': 'Nenhuma imagem recebida'}), 400

    imagens = request.files.getlist('image')
    indice = request.form.get('indice')

    resultados = []
    for file in imagens:
        if file.filename == '':
            continue
        try:
            imagem = Image.open(file.stream).convert('RGB')
            resultado = analisar_folha(imagem, indice)
            resultado['nome_arquivo'] = file.filename
            resultados.append(resultado)
        except Exception as e:
            resultados.append({'erro': f'Erro ao processar imagem {file.filename}: {str(e)}'})

    return jsonify(resultados)

def abrir_navegador():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == '__main__':
    threading.Timer(1.5, abrir_navegador).start()
    app.run(host="0.0.0.0", port=10000)





