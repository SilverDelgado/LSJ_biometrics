
import face_recognition
import csv
import numpy as np
import os

class SistemaReconocimientoFacial:
    def __init__(self, db_path="bd_biometrica.csv"):
        self.db_path = db_path
        # Inicializa la BD .csv si no existe
        if not os.path.exists(self.db_path):
            with open(self.db_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["id", "template"])

    def _guardar_en_bd(self, id, template):
        """Escribe en la BD. Convierte el array numpy a string para el CSV."""
        with open(self.db_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            template_str = ','.join(map(str, template))
            writer.writerow([id, template_str])

    def _leer_bd(self):
        """Lee la BD para el Comparador."""
        ids = []
        templates = []
        with open(self.db_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                ids.append(row["id"])
                # Reconstruye el array desde el CSV
                vector = np.array([float(x) for x in row["template"].split(',')])
                templates.append(vector)
        return ids, templates

    # --- 1. MÉTODOS DE FLUJO COMPLETO ---
    def inscribir(self, id, foto_path):
        """Usa el detector y el extractor[cite: 2, 3]"""
        img = face_recognition.load_image_file(foto_path)
        # El detector localiza la cara y el extractor saca el template[cite: 3]
        encodings = face_recognition.face_encodings(img)
        if encodings:
            self._guardar_en_bd(id, encodings[0])
            return True
        return False

    def verificar(self, id, foto_path):
        """Verifica una persona a partir de su nombre y foto"""
        img = face_recognition.load_image_file(foto_path)
        encodings = face_recognition.face_encodings(img)
        if not encodings: return False
        
        template_foto = encodings[0]
        ids_bd, templates_bd = self._leer_bd()
        
        # El Comparador[cite: 3]
        for i, id_bd in enumerate(ids_bd):
            if id_bd == id:
                coincide = face_recognition.compare_faces([templates_bd[i]], template_foto)[0]
                return coincide
        return False

    def identificar(self, foto_path):
        """Identifica a una persona comparando contra toda la BD"""
        # ... (Lógica similar a verificar pero iterando buscando el match más cercano)
        pass

    # --- 2. MÉTODOS OMITIENDO EL DETECTOR (_sd) ---
    def inscribir_sd(self, id, foto_recortada_path):
        """Omite el detector de caras. Asume que la foto ya es el rostro."""
        img = face_recognition.load_image_file(foto_recortada_path)
        alto, ancho, _ = img.shape
        # Forzamos las coordenadas (top, right, bottom, left) a los bordes de la imagen
        bounding_box = [(0, ancho, alto, 0)]
        
        # Pasa directamente al extractor de características[cite: 3]
        encodings = face_recognition.face_encodings(img, known_face_locations=bounding_box)
        if encodings:
            self._guardar_en_bd(id, encodings[0])