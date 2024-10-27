import pandas as pd
import re
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import skfuzzy as fuzz

#modulo 1: preprocesar el dataset
data=pd.read_csv("/home/cesaralonso/fuzzy-rule-based-sentiment-analysis/test_data.csv",encoding='ISO-8859-1')

#3.1 pre procesado del text
def decontracted(phrase):   # text pre-processing 
        # specific
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        phrase = re.sub(r"@", "" , phrase)         # removal of @
        phrase =  re.sub(r"http\S+", "", phrase)   # removal of URLs
        phrase = re.sub(r"#", "", phrase)          # hashtag processing
    
        # general
        phrase = re.sub(r" t ", " not ", phrase)
        phrase = re.sub(r" re ", " are ", phrase)
        phrase = re.sub(r" s ", " is ", phrase)
        phrase = re.sub(r" d ", " would ", phrase)
        phrase = re.sub(r" ll ", " will ", phrase)
        phrase = re.sub(r" t ", " not ", phrase)
        phrase = re.sub(r" ve ", " have ", phrase)
        phrase = re.sub(r" m ", " am ", phrase)
        return phrase


# Asegurar que el lexicon esté disponible
nltk.download('vader_lexicon')

# Se inicializa el analizador de sentimientos
sia = SentimentIntensityAnalyzer()

#Módulo 3: fuzzification

#3.3.1 Fuzificacion:
#Los resultados del modulo anterior son utilizados para crear la función de membresía triangular

# Se genera el universo de variables
#   * pos and neg on subjective ranges [0, 1]
#   * op has a range of [0, 10] in units of percentage points
x_p = np.arange(0, 1, 0.1)  #x_p positivo entre [0,1]
x_n = np.arange(0, 1, 0.1)  #x_n negativo entre [0,1]
x_op = np.arange(0, 10, 1)  #x_op salida, el rango es [0,10] para todos los lexicons

# se generan las funciones de membresía
p_lo = fuzz.trimf(x_p, [0, 0, 0.5])     #positivo low
p_md = fuzz.trimf(x_p, [0, 0.5, 1])     #positivo medium
p_hi = fuzz.trimf(x_p, [0.5, 1, 1])     #positivo high
n_lo = fuzz.trimf(x_n, [0, 0, 0.5])     #negativo low 
n_md = fuzz.trimf(x_n, [0, 0.5, 1])     #negativo medium
n_hi = fuzz.trimf(x_n, [0.5, 1, 1])     #negativo high

op_Neg = fuzz.trimf(x_op, [0, 0, 5])    # Escala : Negativo Neutral Positivo
op_Neu = fuzz.trimf(x_op, [0, 5, 10])
op_Pos = fuzz.trimf(x_op, [5, 10, 10])


#Módulo 4: Base de reglas
#3.3.2. Se realiza la creación de la base de reglas
# se obtienen de la interecepción de dos entradas
# los valores positivos y negativos cada uno, con tres subconjuntos difusos
# cada data point activa una y solo una regla

def obtener_puntaje_sentimiento(text):
    #Modulo 2: Lexicon de sentimientos
    #3.2 Uso del lexicon de sentimientos
    new_text = decontracted(text)
    #calcular el valor del sentimiento negativo y positivo usando VADER
    ss = sia.polarity_scores(new_text)
    # Extraer los valores positivos y negativos del tweet
    posscore=ss['pos']
    negscore=ss['neg']

    # Se activan las funciones de membresía, opteniendo el grado de pertenencia
    # de cada valor con cada nivel
    # params: 
    # 1: el universo, 2: subconjunto, 3: la variable  positiva o negativa
    p_level_lo = fuzz.interp_membership(x_p, p_lo, posscore)
    p_level_md = fuzz.interp_membership(x_p, p_md, posscore)
    p_level_hi = fuzz.interp_membership(x_p, p_hi, posscore)
    
    n_level_lo = fuzz.interp_membership(x_n, n_lo, negscore)
    n_level_md = fuzz.interp_membership(x_n, n_md, negscore)
    n_level_hi = fuzz.interp_membership(x_n, n_hi, negscore)

    #se devuelve el menor valor de pertenencia entre ambas entradas
    active_rule1 = np.fmin(p_level_lo, n_level_lo)  #(15)
    active_rule2 = np.fmin(p_level_md, n_level_lo)  #(16)
    active_rule3 = np.fmin(p_level_hi, n_level_lo)  #(17)
    active_rule4 = np.fmin(p_level_lo, n_level_md)  #(18)
    active_rule5 = np.fmin(p_level_md, n_level_md)  #(19)
    active_rule6 = np.fmin(p_level_hi, n_level_md)  #(20)
    active_rule7 = np.fmin(p_level_lo, n_level_hi)  #(21)
    active_rule8 = np.fmin(p_level_md, n_level_hi)  #(22)
    active_rule9 = np.fmin(p_level_hi, n_level_hi)  #(23)

        
    #3.3.3 ecuaciones (24,25,27,28,29)
    # ahora optenemos el máximo entre 3 reglas para optener las reglas de salida
    n1=np.fmax(active_rule4,active_rule7)
    n2=np.fmax(n1,active_rule8)             #(24 w_neg) emocion negativa     
    op_activation_lo = np.fmin(n2,op_Neg)   #(27)
    
    neu1=np.fmax(active_rule1,active_rule5)
    neu2=np.fmax(neu1,active_rule9)         #(25 w_neu) emocion neutral
    op_activation_md = np.fmin(neu2,op_Neu) #(28)
    
    p1=np.fmax(active_rule2,active_rule3)
    p2=np.fmax(p1,active_rule6)             #(w_pos) emotion positiva
    op_activation_hi = np.fmin(p2,op_Pos)   #(29)

    # union de los consecuentes para obtener el output (30)
    aggregated = np.fmax(op_activation_lo,
                         np.fmax(op_activation_md, op_activation_hi))
    

    # modulo5: desfuzzificacion
    # desfuzzificar con el método del centroide
    numerador = np.sum(x_op * aggregated)  # Suma de x_op * grado de pertenencia
    denominador = np.sum(aggregated)       # Suma de los grados de pertenencia

    #denominador no sea cero para evitar división por cero
    op = numerador / denominador if denominador != 0 else 0
    output = round(op, 2)

    # rango : Neg Neu Pos   
    # Clasificación de sentimiento
    if 0 < output < 3.33:
        sentiment = "Negative"
    elif 3.33 < output < 6.67:
        sentiment = "Neutral"
    elif 6.67 < output < 10:
        sentiment = "Positive"
    
    # retorna puntajes de sentimiento y la clasificación final
    return posscore, negscore, sentiment


# Llamada a obtener_puntaje_sentimiento para cada fila, almacenando los resultados en una lista
resultados = data['sentence'].apply(obtener_puntaje_sentimiento).tolist()

# Convertimos la lista de resultados en un DataFrame y asignamos las tres columnas al DataFrame original
resultados_df = pd.DataFrame(resultados, columns=['TweetPos', 'TweetNeg', 'New_Sentiment'], index=data.index)

# Asignamos el DataFrame al original
data[['TweetPos', 'TweetNeg', 'New_Sentiment']] = resultados_df

# Se guarda el resultado en un nuevo CSV
print("Se escribe el resultado...")
output_file_path = '/home/cesaralonso/fuzzy-rule-based-sentiment-analysis/analisis_resultado.csv'
data.to_csv(output_file_path, index=False)