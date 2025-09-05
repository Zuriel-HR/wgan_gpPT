import os
import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image
import xml.etree.ElementTree as ET

import json
import time
from datetime import datetime


#-----------------------------

# Esta función se encarga de leer un archivo XML específico y retornar las funciones que contiene.
# Útil para el filtrado de funciones al generar archivos JSON.
def extraer_funciones_desde_xml(xml_path, output_path="funciones_extraidas.txt"):
    """
    Extrae funciones desde un archivo XML y las guarda ordenadas alfabéticamente en un archivo de texto.

    Parámetros:
        xml_path (str): Ruta al archivo XML de entrada.
        output_path (str): Ruta al archivo de salida. Por defecto, 'funciones_extraidas.txt'.
    Formato de XML:

    Retorna:
        list: Lista de funciones extraídas en formato 'librería::función'.
    """
    # Parsear el XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Lista para almacenar funciones
    funciones = []

    # Recorremos todas las librerías
    for lib in root.findall(".//lib"):
        libname = lib.get("name")
        fcts = lib.find("fcts")
        if fcts is not None:
            for fct in fcts.findall("fct"):
                if fct.text is not None:
                    function_name = fct.text.strip()
                    funciones.append(f"{libname}::{function_name}")

    # Ordenar alfabéticamente
    #funciones = sorted(funciones)

    # Guardar en archivo
    #with open(output_path, "w", encoding="utf-8") as f:
    #    for func in funciones:
    #        f.write(func + "\n")

    print(f"Total de funciones extraídas: {len(funciones)}")
    return funciones
# Esta función es para obtener en una variable las lib:funcion existentes en las muestras de una archivo txt.
def extraer_funciones_desde_txt(txt_path="./Extraer_Caracteristicas/funciones_existentes.txt"):
    with open(txt_path, "r") as f:
        lista_funciones = [linea.strip() for linea in f]  # .strip() quita el salto de línea.
    return lista_funciones
# --- Parámetros globales ---
xml_path = "./Extraer_Caracteristicas/functions.xml"
NORMALIZAR_ENTRADA = "0-1"  # opciones: "0-1" o "menos1-a-1"
FUNCIONES_API = extraer_funciones_desde_txt(txt_path="./data/funciones_existentes.txt")# extraer_funciones_desde_xml(xml_path=xml_path)
print("longitud de FUNCIONES_API: ", len(FUNCIONES_API))
# Agregamos contador a la lista.
func_to_index = {func: idx for idx, func in enumerate(FUNCIONES_API)}
# Este número se utilizará al normalizar valores de json a vector.
N = len(FUNCIONES_API)

# Esta función sirve para normalizar los valores al convertir de JSON a NPy.
def normalizar(valor, min_val, max_val):
    # Controlamos los límites.
    if valor < min_val:
        valor = min_val
    elif valor > max_val:
        valor = max_val
    # Normalizamos valores de 0 a 1.
    norm = (valor - min_val) / (max_val - min_val)
    # Normalizamos valores de -1 a 1, dependiendo de la variable global.
    if NORMALIZAR_ENTRADA == "menos1-a-1":
        return norm * 2 - 1
    return norm

def vector_a_imagen(vector, size=(64, 64)):
    total_pixeles = size[0] * size[1]
    if len(vector) < total_pixeles:
        vector = np.pad(vector, (0, total_pixeles - len(vector)))
    elif len(vector) > total_pixeles:
        vector = vector[:total_pixeles]
    imagen = vector.reshape(size)
    imagen = (imagen * 2) - 1  # Normaliza de [0,1] a [-1,1]
    return imagen.astype(np.float32)

# Esta función se encarga de procesar un JSON y convertirlo a formato npy (para una red neuronal).
def json_a_vector(ruta_json):
    # Abrimos el archivo JSON pasado por parámetros.
    with open(ruta_json, "r", encoding="utf-8") as f:
        try:
            # Cargamos los datos de archivo.
            datos = json.load(f)
            # nombre de familia.
            nombres = [
    "benign", "_Action", "_Chthonic", "_Citadel", "_Evo-Zeus (Evolution)", "_Flokibot",
    "_Grabbot", "_Ice IX", "_KINS", "_Murofet", "_Pandabanker", "_PowerZeus",
    "_Prog", "_Satan", "_Skynet", "_Tasks", "_Uncategorized",
    "_Unnamed 1", "_Unnamed 10", "_Unnamed 2", "_Unnamed 3", "_Unnamed 4",
    "_Unnamed 7", "_Unnamed 8", "_Unnamed 9", "_VMZeus", "_ZeuS 1",
    "_ZeuS 2", "_ZeuS v4 (Unnamed 5)", "_Zeus z2.5 (Unnamed 6)",
    "_ZeusAES", "_ZeusX", "_Zloader 2 (Silent Night)", "_Zloader", "ZeuS Over Tor (Sphinx)", "ZeuS-P2P (GameOver)"
]           
            ordenados = sorted(nombres)
            fam_name = datos.get("file_info", {}).get("software_family", "").replace(" ", "")
            # Normalizamos el nombre de familia de malware
            fam_name_norm = (ordenados.index(fam_name) * 2 + 1) / (2 * len(ordenados)) if fam_name in ordenados else 0.0
            fam_name_norm = normalizar(fam_name_norm, 0, 1)
            
            # file_size en formato X[MB|KB].
            size_str = datos.get("file_info", {}).get("file_size", "0KB").upper().replace(" ", "")
            size_bytes = 0
            # Revisamos que tipo de dato tiene para hacer la conversión a bits.
            if "MB" in size_str:
                size_bytes = float(size_str.replace("MB", "")) * 1024 * 1024
            elif "KB" in size_str:
                size_bytes = float(size_str.replace("KB", "")) * 1024 
            elif size_str.isdigit(): # Si solo son números
                size_bytes = int(size_str)
            # Normalizamos con valor y sus límites [0,50MB].
            size_norm = normalizar(size_bytes, 0, 50 * 1024 * 1024)


            # timestamp en formato "Fri Jan 16 09:03:59 2015\n"
            time_str = datos.get("analysis", {}).get("compilation_timestamp", "").strip()
            try:
                # Interpretamos y convertimos a formato datetime y a struct_time: díaSem, mes, día, hora, año.
                timestamp = time.mktime(datetime.strptime(time_str, "%a %b %d %H:%M:%S %Y").timetuple())
            except:
                timestamp = 0
            # El rango de fechas se estableció entre el 2005 al 2030.
            # timestamp ahora significará los segundos desde 1970-01-01 en la zona horaria local.
            primerDia2005 =  1104537600
            primerDia2030 = 1893456000
            timestamp_norm = normalizar(timestamp, primerDia2005, primerDia2030)


            # entropy
            entropy = datos.get("analysis", {}).get("entropy", 0)
            entropy_norm = normalizar(entropy, 0, 8)


            # sections
            # Obtenemos la lista de secciones en diccionario con clave <Nombre> y valor <Resto_de_sección>
            # ubicada en analysis > sections dentro del JSON.
            secciones = {s["name"]: s for s in datos.get("analysis", {}).get("sections", [])}
            orden_secciones = [".text", ".idata", ".rsrc", ".data"]
            section_vector = []
            for nombre in orden_secciones:
                # Revisamos si existe "nombre" en la lista de secciones.
                if any(seccion.get("name") == nombre for seccion in secciones.values()):
                    s = secciones.get(nombre, {})
                    ent = normalizar(s.get("entropy", 0), 0, 8)
                    v_size = normalizar(s.get("virtual_size", 0), 0, 1_000_000)
                    r_size = normalizar(s.get("raw_size", 0), 0, 1_000_000)
                    
                else: # No existe sección con este nombre.
                    # Establecemos los valores de sección en 0 para la normalización.
                    ent = normalizar(0, 0, 8)
                    v_size = normalizar(0, 0, 1_000_000)
                    r_size = normalizar(0, 0, 1_000_000)
                # Actualizamos el vector.
                section_vector.extend([ent, v_size, r_size])

            # Buscar secciones que no están en orden_secciones
            otras_secciones = [s for nombre, s in secciones.items() if nombre not in orden_secciones]
            if otras_secciones: # Verificar si hay alguna sección adicional
                seccion_mayor_entropia = max(otras_secciones, key=lambda s: s.get("entropy", 0)) # Buscar la sección con mayor entropía
                # Obtener y normalizar sus valores
                ent = normalizar(seccion_mayor_entropia.get("entropy", 0), 0, 8)
                v_size = normalizar(seccion_mayor_entropia.get("virtual_size", 0), 0, 1_000_000)
                r_size = normalizar(seccion_mayor_entropia.get("raw_size", 0), 0, 1_000_000)
                # Agregar al vector
                section_vector.extend([ent, v_size, r_size])
            else:
                # Si no hay secciones adicionales, agregar ceros
                section_vector.extend([
                    normalizar(0, 0, 8),
                    normalizar(0, 0, 1_000_000),
                    normalizar(0, 0, 1_000_000)
                ])

            # imports
            funciones_encontradas = set()
            # Obtenemos la lista de dll importados.
            dlls = datos.get("analysis", {}).get("imports", {}).get("dlls", [])
            for dll in dlls:
                # Nombre sin espacios ni minúsculas.
                lib_name = dll.get("libname", "").strip().lower()
                # Lista de funciones.
                funciones = dll.get("functions", [])
                for func in funciones:
                    nombre_completo = f"{lib_name}::{func.strip()}"
                    # Y agregamos el formato > lib_name::func a funciones_encontradas.
                    funciones_encontradas.add(nombre_completo)
            # Creamos el vector tomando el total de funciones trackeadas.
            vector_imports = np.zeros(N, dtype=int)
            # Revisamos si alguna función encontrada está en la lista de funciones (enumerada).
            # Quedará normalizado de 0 a 1.
            for func in funciones_encontradas:
                    if func in func_to_index:
                        vector_imports[func_to_index[func]] = 1
            # Revisamos si es necesario normalizar de -1 a 1.
            if NORMALIZAR_ENTRADA == "menos1-a-1":
                vector_imports = vector_imports * 2 - 1


            # vector final convertido a float32
            vector_datos = np.array(
                [fam_name_norm, size_norm, timestamp_norm, entropy_norm] + section_vector,
                dtype=np.float32
            )
            # Todos normalizados respecto a la variable NORMALIZAR_ENTRADA.
            vector_completo = np.concatenate([vector_datos, vector_imports])
            return vector_completo
            # imagen
            #imagen = vector_a_imagen(vector_completo)
            #imagen = np.expand_dims(imagen, axis=-1)
            #np.save(ruta_img, imagen)
            #np.save(ruta_img, vector_completo)
#
            #BLUE = '\033[94m'
            #RESET = '\033[0m'
            #print(BLUE + f"Vector guardado: {ruta_img}" + RESET)

        except json.JSONDecodeError:
            print(f"Error leyendo el archivo JSON: {ruta_json}")

# Esta función se encarga de llevar un registro del los tamaños en bits de los archivos.
def agregar_numero_a_archivo(ruta_archivo, numero):
    try:
        # Abre el archivo en modo de escritura (append), para agregar contenido al final
        with open(ruta_archivo, 'a') as archivo:
            # Agrega el número en una nueva línea
            archivo.write(f"{numero}\n")
        print(f"Se ha agregado el número {numero} al archivo {ruta_archivo}.")
    except Exception as e:
        print(f"Ocurrió un error al intentar escribir en el archivo: {e}")

def guardar_imagen_con_pillow(ruta_img_completa, image_array):
    # Asegurarse de que el array tiene un rango de 0 a 255 y convertir a tipo uint8
    image_array = (image_array * 255).astype(np.uint8)
    
    # Crear una imagen de Pillow a partir del array de NumPy
    img = Image.fromarray(image_array)
    
    # Guardar la imagen en escala de grises (modo 'L')
    img.save(ruta_img_completa)

def json_a_img(ruta_json, ruta_folder_img, width=32):
    try:
        
        # Abrimos y leemos el archivo json.
        with open(ruta_json, 'rb') as f:
            contenido = f.read()
        # Convertimos a un array de 8 bits (valores de 0 a 255)
        byte_array = np.frombuffer(contenido, dtype=np.uint8)

        # Guardamos el número de bytes en un archivo 
        ruta = "./Resultados/2 json/tam_bytes_arrays.txt"
        numero_a_agregar = byte_array.nbytes
        agregar_numero_a_archivo(ruta, numero_a_agregar)

        # Calculamos la altura de la imagen
        height = len(byte_array) // width
        # Recortamos el byte_array si no es divisible por el ancho para evitar errores
        byte_array = byte_array[:height * width]
        # Redimensionar el vector de 1D a 2D (imagen en escala de grises)
        image = byte_array.reshape((height, width))

        # Mostramos la imagen
        #plt.imshow(image, cmap='gray')
        #plt.title("Malware json as Image")
        #plt.show()
        nombre_con_extension = os.path.split(ruta_json)[1]
        nombre_sin_extension = os.path.splitext(nombre_con_extension)[0]
        ruta_img_completa = os.path.join(ruta_folder_img, nombre_sin_extension + '.png')
        # Guardar la imagen
        guardar_imagen_con_pillow(ruta_img_completa, image)
        #plt.imsave(ruta_img_completa, image, cmap='gray')
        
        print(f"Archivo img creado: '{ruta_img_completa}'.")
    
    except FileNotFoundError:
        print(f"Error: El archivo '{ruta_json}' no se encuentra.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")
