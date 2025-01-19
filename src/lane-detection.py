import cv2
import numpy as np  
import math

""" Threshold: Umbral de las luces grises o negras 
    kernell:  minimiza ruido
    ROI:      Region de interes, mientras mas a la derecha, mas abajo en la imagen
"""

#------------------------------------------------------------------------------------------------

# Función para clasificar líneas detectadas en categorías: líneas inclinadas hacia la izquierda,
# líneas inclinadas hacia la derecha y líneas horizontales. 
# Se clasifica según el ángulo de la pendiente.

#para todas las lineas quiero calcular la pendiente, segun el valor del angulo, es una linea derecha o izquierda
def lines_classifier(lines):
    left_lines = []    
    right_lines = []
    horizontal_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]  # Coordenadas de los extremos de la línea
            if x2 - x1 == 0:  # Si la línea es vertical (división por 0)
                slope = np.pi / 2
            else:
                slope = np.arctan((y2 - y1) / (x2 - x1))  # Cálculo de la pendiente
            angle_degrees = np.degrees(abs(slope))  # Conversión del ángulo a grados
            # Clasificación según el ángulo
            if abs(angle_degrees) > 20 and (angle_degrees < 160 or angle_degrees > 200): #si es menor a 20° la consideramos de una forma, podemos hacer otra funcion para detectar lineas.
                if slope < 0:
                    left_lines.append(line)  # Líneas inclinadas hacia la izquierda
                else:
                    right_lines.append(line)  # Líneas inclinadas hacia la derecha
            elif angle_degrees < 10 or (170 < angle_degrees < 190):
                if y1 == y2:  # Detectar líneas horizontales largas
                    long = math.sqrt((x2 - x1) * 2 + (y2 - y1) * 2)  # Longitud de la línea
                    if long > 100: 
                        horizontal_lines.append(line)
                    
    return left_lines, right_lines, horizontal_lines

#-------------------------------------------------------------------------------------------------

# Función para calcular el promedio de varias líneas. Si hay varias líneas en la lista, 
# devuelve la línea promedio (como la media de los extremos).

def average_lines(lines):
    if len(lines) > 0:
        lines_array = np.array(lines)  # Convertir la lista de líneas en un array de Numpy
        average_line = np.mean(lines_array, axis=0, dtype=np.int32)  # Calcular el promedio
        return average_line
    else:
        return None
    
#-------------------------------------------------------------------------------------------------

# Función para calcular el punto de intersección de una línea con una coordenada y determinada.
# Calcula el punto donde una línea dada intersecta la línea y = const.

def get_intersection_point(line, y):
    x1, y1, x2, y2 = line[0]  # Coordenadas de los extremos de la línea
    slope = (y2 - y1) / (x2 - x1)  # Calcular la pendiente
    if slope == 0:
        return x1, y  # Si la pendiente es 0 (línea horizontal), devolver el x original
    else:
        x = int(x1 + (y - y1) / slope)  # Calcular el valor de x correspondiente
        return x, y

#-------------------------------------------------------------------------------------------------

# Función para dibujar una línea extendida sobre la imagen. La línea se dibuja desde su intersección
# con el borde superior hasta su intersección con el borde inferior de la imagen.

def line_drawing(line, height):
    extended_line = np.array([
            get_intersection_point(line, 0),  # Intersección con el borde superior (y = 0)
            get_intersection_point(line, height)  # Intersección con el borde inferior (altura)
        ], dtype=np.int32)
    # Dibuja la línea extendida en la imagen con color rojo y grosor 2
    cv2.line(frame, (extended_line[0][0], extended_line[0][1]), 
             (extended_line[1][0], extended_line[1][1]), (0, 0, 255), 2)

#-------------------------------------------------------------------------------------------------------------------

# Función para fusionar líneas que están cerca unas de otras. Si una línea está cerca de una línea
# ya fusionada, se une con ella.
#para evitar tener varias lineas juntas en un mismo lugar o cercanas

def merge_lines(lines):
    merged_lines = []
    for line in lines:
        if len(merged_lines) == 0:
            merged_lines.append(line)
        else:
            # Comprobar si la línea actual está lo suficientemente cerca de alguna de las líneas fusionadas
            merge_flag = False
            for i, merged_line in enumerate(merged_lines):
                if abs(line[0][0] - merged_line[0][2]) < 75:  # Compara la distancia entre extremos de las líneas
                    # Fusionar líneas si están suficientemente cerca
                    merged_lines[i] = np.array([[merged_line[0][0], merged_line[0][1], line[0][2], line[0][3]]])
                    merge_flag = True
                    break
            if not merge_flag:
                merged_lines.append(line)
    return merged_lines

#-------------------------------------------------------------------------------------------------------------------

# Función para calcular el error midiendo el punto medio entre las líneas izquierda y derecha.

def getting_error(average_left_line, average_right_line, height, width):
    # Verifica si las líneas izquierda y derecha no son None
    if average_left_line is not None and average_right_line is not None:
        # Descomposición de las líneas promedio
        x1_left, y1_left, x2_left, y2_left = average_left_line[0]
        x1_right, y1_right, x2_right, y2_right = average_right_line[0]

        # Calcular los puntos donde las líneas promedio izquierda y derecha intersectan el borde inferior de la imagen
        bottom_left_x = int(x1_left + (height - y1_left) * (x2_left - x1_left) / (y2_left - y1_left))
        bottom_right_x = int(x1_right + (height - y1_right) * (x2_right - x1_right) / (y2_right - y1_right))

        # Calcular el punto medio entre estos puntos
        midpoint_x = (bottom_left_x + bottom_right_x) // 2
        midpoint_y = height

        # Asegurarse de que no haya división por cero al calcular la pendiente
        if x2_left - x1_left == 0:
            x2_left = x1_left + 0.01
        if x2_right - x1_right == 0:
            x2_right = x1_right + 0.01

        # Calcular la pendiente de las líneas
        slope_left = (y2_left - y1_left) / (x2_left - x1_left)
        slope_right = (y2_right - y1_right) / (x2_right - x1_right)

        # Manejar el caso cuando las líneas son paralelas
        if slope_left == slope_right:
            slope_right += 0.01

        # Calcular la intersección de las líneas promedio izquierda y derecha
        intersection_x = int((y1_right - y1_left + slope_left * x1_left - slope_right * x1_right) / (slope_left - slope_right))
        intersection_y = int(slope_left * (intersection_x - x1_left) + y1_left)

        # Dibujar puntos y líneas en la imagen (opcional)
        cv2.circle(frame, (intersection_x, intersection_y), 5, (255, 0, 255), -1)
        cv2.circle(frame, (midpoint_x, midpoint_y), 5, (0, 255, 255), -1)
        cv2.line(frame, (intersection_x, intersection_y), (midpoint_x, midpoint_y), (0, 165, 255), 2)

        bottom_center_x = width // 2
        bottom_center_y = height
        cv2.circle(frame, (bottom_center_x, bottom_center_y), 5, (0, 255, 0), -1)

        point_A = int(midpoint_x)
        point_B = int(bottom_center_x)
        point_C = int(intersection_x + (0 - intersection_y) * (midpoint_x - intersection_x) / (midpoint_y - intersection_y))

        cv2.circle(frame, (point_C, width), 5, (255, 255, 0), -1)
    else:
        print("Error: average_left_line o average_right_line es None")
    
    return

#---------------------------------------------------------------------------------------------------------

# Función principal de procesamiento de imagen. Aplica filtros, detecta líneas con Canny, 
# las clasifica, las fusiona y dibuja las líneas detectadas.

def image_processing(image):
    # Obtiene los valores de los trackbars para ajustar los parámetros en tiempo real
    threshold_value = cv2.getTrackbarPos('Threshold', 'Canny')
    kernel_value = cv2.getTrackbarPos('Valor Kernel', 'Canny') | 1  # Asegura que sea impar, segun como limpias imagenes, hay muchas formas, podemos ver que nos conviene
    ROI_value = cv2.getTrackbarPos('Valor ROI', 'Canny')/100  # ROI como porcentaje de la altura, Region de interes. nos da un valor entre 0 y 1

    height, width = image.shape[:2]  # Obtiene las dimensiones de la imagen. saco el alto y el ancho

    # Define el área de interés (ROI) como un polígono. 
    x1 = 0
    x2 = width
    y1 = int(ROI_value * height) 
    y2 = height

    #definimos la region de interes, indicamos las cordenadas de cuales son nuestra zona de interes. Arranca a medir desde arriba izq (0,0). y para abajo e x para derecha
    #si aplico un roivalue mas alto o mas bajo nos da un area mas grande o chica
    #mientras mas grande el roi (va de abajo para arriba) mas pantalla capturo
    roi_vertices = np.array([[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]])    
    mask = np.zeros_like(image)  # Crea una máscara negra del tamaño de la imagen. Imagen con todos 0, el mismo color
    cv2.fillPoly(mask, roi_vertices, (255, 255, 255))  # Rellena el ROI con blanco
    masked_image = cv2.bitwise_and(image, mask)  # Aplica la máscara a la imagen. Muestra la imagen solo en la zona de interes.

    grey_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)  # Convierte la imagen a escala de grises
    # Ahora vamos a Aplicar umbral binario, escala de grises, con extremos de blancos y negros. Segun el threshold. Si le pongo 125, 
    # si el valor del pixel es mayor es ese numero, se va a identificar de un color:
    _, binary_image = cv2.threshold(grey_image, threshold_value, 255, cv2.THRESH_BINARY)  
    noiseless_image = cv2.medianBlur(binary_image, kernel_value)  # Aplica filtro de mediana para reducir ruido. Kernel es el parametro. 
                                        #Tiene que ser impar. Le dice, sacame los puntos que no me interesan, sacame los pixel que no comparten informacion
    canny_image = cv2.Canny(noiseless_image, 100, 150)  # Detecta bordes usando Canny

    # Detecta líneas usando la Transformada de Hough, buscamos la lineas con contornos, nos detecta lineas. 
    lines = cv2.HoughLinesP(canny_image, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=1500) #un poco es, cantidad de pixeles, longitud que tiene que tener el contorno para poder decir que es una linea
                                                                            #se puede agregar como parametro para despues jugar y ver que tan facil reconoce lineas

    # Clasifica las líneas en inclinadas a la izquierda, derecha y horizontales
    left_lines, right_lines, horizontal_lines = lines_classifier(lines) #para todas las lineas quiero calcular la pendiente, segun el valor del angulo, es una linea derecha o izquierda

    # Fusiona las líneas clasificadas y calcula las líneas promedio
    merged_left_lines = merge_lines(left_lines)
    merged_right_lines = merge_lines(right_lines)
    merged_horizontal_lines = merge_lines(horizontal_lines)

    average_left_line = average_lines(merged_left_lines) #me calcula una linea promedio en cada lado, izq y derecha
    if average_left_line is not None:
        line_drawing(average_left_line, height = height)  # Dibuja la línea izquierda promedio

    average_right_line = average_lines(merged_right_lines)
    if average_right_line is not None:
        line_drawing(average_right_line, height = height)  # Dibuja la línea derecha promedio
    
    average_horizontal_line = average_lines(merged_horizontal_lines)
    if average_horizontal_line is not None:
        x1, y1, x2, y2 = average_horizontal_line[0]
        long = math.sqrt((x2 - x1) * 2 + (y2 - y1) * 2)
        if long > 100:
            cv2.line(image, (x1, y1), (x2, y2), (255, 130, 0), 2)  # Dibuja la línea horizontal promedio

    return average_left_line, average_right_line, height, width, canny_image

#------------------------------------------------------------------------------------------
# Configuración de ventanas y trackbars para ajustar los parámetros en tiempo real

cv2.namedWindow('Canny') 
cv2.namedWindow('Camera Stream')

camera = cv2.VideoCapture("http://192.168.0.101/stream")

# Creación de trackbars para ajustar el umbral, el valor del kernel y el valor de ROI
#no se hace con el auto en movimiento, esto lo usamos para variar los parametros segun la iluminacion que haya en el lugar
#es mas facil crear un deslizador para ir ajustando y ir viendo donde funciona mejor
#es buena idea que el auto pued     e variar esos parametros automaticamente, asi puede detectar mejor

cv2.createTrackbar('Threshold', 'Canny', 0, 250, lambda x: None)# (el tipo de parametro, ventana, minimo, maximo, y variable)
cv2.createTrackbar('Valor Kernel', 'Canny', 0, 10, lambda x: None)
cv2.createTrackbar('Valor ROI', 'Canny', 0, 100, lambda x: None)

while True: 
    ret, frame = camera.read() #obtenemos 2 cosas, el ret es true/false

    #si no logramos ver nada, cortamos el codigo
    if not ret:
        print("Error al capturar la imagen desde la cámara.")
        break

    # Procesa la imagen capturada
    average_left_line, average_right_line, height, width, canny_image = image_processing(frame)
    getting_error(average_left_line, average_right_line, height, width) # le paso lineas izq y derecha, y hace la linea roja en el medio, es con lo que arrancamos a hacer un controlador PID del sistema de dirección

    # Muestra la imagen original y la salida de Canny
    cv2.imshow('Camera Stream', frame)
    cv2.imshow('Canny', canny_image) #imagen en black/white

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
camera.release()
cv2.destroyAllWindows()