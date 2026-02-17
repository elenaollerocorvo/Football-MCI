# Listing A.1 : Labeling code
# Script para etiquetar videos/audios de fútbol mediante interfaz gráfica
# NO ESTABA INCLUIDO EN EL CODIGO
# se crea a partir del fichero PDF.

# Importación de bibliotecas necesarias para la interfaz gráfica
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QLineEdit,
QPushButton, QComboBox, QVBoxLayout, QWidget, QFileDialog, QFrame,
QMessageBox, QSlider, QHBoxLayout, QStyle)
from PyQt6.QtGui import *
from PyQt6.QtCore import QStandardPaths, Qt, QUrl, pyqtSlot, pyqtSignal
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput, QVideoSink
from PyQt6.QtMultimediaWidgets import QVideoWidget
import cv2  # OpenCV para procesamiento de video
import time
import os

import json  # Para guardar y cargar configuraciones
import sys

class FirstWindow(QMainWindow):
    """
    Ventana principal de configuración del proyecto de etiquetado.
    Permite configurar nombre del proyecto, método de etiquetado, clases y base de decisión.
    """
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """Inicializa la interfaz de usuario con todos los campos de configuración"""
        # Set the window title
        self.setWindowTitle('First Window')

        # Main layout
        main_layout = QVBoxLayout()

        # Labeling Project Name
        self.label_name = QLabel("Name of the Labeling Project:")
        main_layout.addWidget(self.label_name)

        self.entry_name = QLineEdit()
        main_layout.addWidget(self.entry_name)

        # Person labeling
        self.label_person = QLabel("Person Labelling:")
        main_layout.addWidget(self.label_person)

        self.entry_person = QLineEdit()
        main_layout.addWidget(self.entry_person)

        # Labeling Method
        self.label_method = QLabel("Labelling Method:")
        main_layout.addWidget(self.label_method)

        self.method_options = ["One-hot-encoding", "Sequences"]
        self.dropdown_method = QComboBox()
        self.dropdown_method.addItems(self.method_options)
        self.dropdown_method.setCurrentIndex(0) # default value
        main_layout.addWidget(self.dropdown_method)

        #Sequence width add
        self.label_sequence_width = QLabel("Sequence Width:")
        self.label_sequence_width.hide() # initially hidden
        main_layout.addWidget(self.label_sequence_width)

        self.entry_sequence_width = QLineEdit()
        self.entry_sequence_width.setValidator(QIntValidator())
        self.entry_sequence_width.hide() # initially hidden
        main_layout.addWidget(self.entry_sequence_width)

        self.dropdown_method.currentIndexChanged.connect(self.on_method_changed)

        # Classes to be labeled
        self.label_classes = QLabel("Classes to be labeled (comma-separated):")
        main_layout.addWidget(self.label_classes)

        self.entry_classes = QLineEdit()
        main_layout.addWidget(self.entry_classes)

        # Decision Basis
        self.label_basis = QLabel("Basis of Decision:")
        main_layout.addWidget(self.label_basis)

        self.basis_options = ["Video", "Audio", "Intuition", "Folder"]
        self.dropdown_basis = QComboBox()
        self.dropdown_basis.addItems(self.basis_options)
        self.dropdown_basis.setCurrentIndex(0) # default value
        main_layout.addWidget(self.dropdown_basis)

        # Buttons layout
        buttons_layout = QHBoxLayout()

        # Load settings button
        self.button_load = QPushButton("Load Settings")
        self.button_load.clicked.connect(self.load_settings)
        buttons_layout.addWidget(self.button_load)

        # Save settings button
        self.button_save = QPushButton("Save Settings")
        self.button_save.clicked.connect(self.save_settings)
        buttons_layout.addWidget(self.button_save)

        # Confirm button
        self.button_confirm = QPushButton("Confirm and View")
        self.button_confirm.clicked.connect(self.open_second_window)
        main_layout.addWidget(self.button_confirm)

        # Add buttons layout to main layout
        main_layout.addLayout(buttons_layout)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)

        self.setCentralWidget(central_widget)

    def open_second_window(self):
        """
        Abre la ventana secundaria de etiquetado con la configuración actual.
        Recopila todos los parámetros y los pasa a SecondWindow.
        """
        # Recopilar valores de los campos de entrada
        project_name = self.entry_name.text()
        labeler = self.entry_person.text()
        method = self.dropdown_method.currentText()
        classes = self.entry_classes.text().split(',') # assuming comma-separated
        decision_basis = self.dropdown_basis.currentText()
        sequence_width = self.entry_sequence_width.text()

        # Crear y mostrar la ventana de etiquetado
        self.second_window = SecondWindow(project_name, labeler, method, classes
        , decision_basis, sequence_width)
        self.second_window.show()
        self.close()


    def load_settings(self):
        """Carga la configuración desde un archivo JSON"""
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Settings",
        QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation), "JSON files (*.json);;All files (*)")

        if filepath:
            with open(filepath, 'r') as file:
                data = json.load(file)

            # Clear the fields
            self.entry_name.clear()
            self.entry_person.clear()
            self.dropdown_method.setCurrentIndex(-1)
            self.entry_classes.clear()
            self.dropdown_basis.setCurrentIndex(-1)
            self.entry_sequence_width.clear() # Clear the sequence width field

            # Populate the fields with loaded data
            self.entry_name.setText(data.get('project_name', ''))
            self.entry_person.setText(data.get('labeler', ''))
            method_index = self.dropdown_method.findText(data.get('method', ''))
            if method_index != -1:
                self.dropdown_method.setCurrentIndex(method_index)
                self.on_method_changed(method_index) # Trigger the visibility
                # change based on loaded method
            self.entry_classes.setText(data.get('classes', ''))
            basis_index = self.dropdown_basis.findText(data.get('decision_basis', ''))
            if basis_index != -1:
                self.dropdown_basis.setCurrentIndex(basis_index)
            self.entry_sequence_width.setText(data.get('sequence_width', ''))
            # Populate the sequence width field

            QMessageBox.information(self, "Load Settings", "Settings loaded successfully!")


    def save_settings(self):
        """Guarda la configuración actual en un archivo JSON"""
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Settings",
        QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation), "JSON files (*.json);;All files (*)")

        if filepath:
            # Crear diccionario con la configuración actual
            data = {
            #'project_name': self.entry_name.text(),
            #'labeler': self.entry_person.text(),
            #'method': self.dropdown_method.currentText(),
            'sequence_width': self.entry_sequence_width.text(),
            'classes': self.entry_classes.text(),
            #'decision_basis': self.dropdown_basis.currentText(),
            }
            with open(filepath, 'w') as file:
                json.dump(data, file, indent=4)
            QMessageBox.information(self, "Save Settings", "Settings saved successfully!")

    def on_method_changed(self, index):
        """
        Callback cuando cambia el método de etiquetado.
        Muestra/oculta el campo de ancho de secuencia según el método seleccionado.
        """
        method = self.dropdown_method.itemText(index)
        if method == "Sequences":
            self.label_sequence_width.show()
            self.entry_sequence_width.show()
        else:
            self.label_sequence_width.hide()
            self.entry_sequence_width.hide()

class MediaPlayerWidget(QWidget):
    """
    Widget reproductor de medios (video/audio) para el etiquetado.
    Permite reproducir archivos individuales o procesar carpetas completas.
    """
    videoFileSelected = pyqtSignal(str)  # Señal emitida cuando se selecciona un video

    def __init__(self, decision_basis, second_window, parent=None):
        """
        Inicializa el reproductor de medios.
        
        Args:
            decision_basis: Tipo de medio ('Video', 'Audio' o 'Folder')
            second_window: Referencia a la ventana principal de etiquetado
            parent: Widget padre (opcional)
        """
        super().__init__(parent)
        # Variables de control del reproductor
        self.duration = 0  # Duración total del medio
        self.mediaPlayer = QMediaPlayer(self)  # Reproductor multimedia
        self.audioOutput = QAudioOutput(self)  # Salida de audio
        self.mediaPlayer.setAudioOutput(self.audioOutput)
        self.audioOutput.setVolume(100)
        
        # Variables para gestión de múltiples videos
        self.video_files = []  # Lista de archivos de video
        self.current_video_index = 0  # Índice del video actual
        self.save_location_folder = ''  # Carpeta donde guardar resultados
        self.video_names = []  # Nombres de los videos
        self.second_window = second_window  # Referencia a ventana principal
        self.video_counter = self.current_video_index
        self.current_media_path = None  # Ruta del archivo de medio actual

        layout = QVBoxLayout(self)

        if decision_basis == 'Video':
            self.videoWidget = QVideoWidget(self)
            self.mediaPlayer.setVideoOutput(self.videoWidget)
            self.videoWidget.setFixedSize(640*2//3, 480*2//3)
            layout.addWidget(self.videoWidget)

            self.load_video_btn = QPushButton("Select Video", self)
            self.load_video_btn.clicked.connect(self.openMediaFile)
            layout.addWidget(self.load_video_btn)

        elif decision_basis == 'Audio':
            self.load_audio_btn = QPushButton("Select Audio", self)
            self.load_audio_btn.clicked.connect(self.openMediaFile)
            layout.addWidget(self.load_audio_btn)

        elif decision_basis == 'Folder':
            self.videoWidget = QVideoWidget(self)
            self.mediaPlayer.setVideoOutput(self.videoWidget)
            self.videoWidget.setFixedSize(640*2//3, 480*2//3)
            layout.addWidget(self.videoWidget)

            self.load_folder_btn = QPushButton("Select Folder", self)
            self.load_folder_btn.clicked.connect(self.openMediaFile)
            layout.addWidget(self.load_folder_btn)

            self.next_video_btn = QPushButton("Next Video", self)
            self.next_video_btn.clicked.connect(self.nextVideo)
            layout.addWidget(self.next_video_btn)

        # Controls layout
        controlsLayout = QHBoxLayout()

        # Play/Pause button
        self.playButton = QPushButton(self)
        self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)
        controlsLayout.addWidget(self.playButton)

        # Playback speed button
        self.speedButton = QPushButton("1.0x", self)
        self.speedButton.clicked.connect(self.changePlaybackSpeed)
        controlsLayout.addWidget(self.speedButton)

        # Progress slider
        self.positionSlider = QSlider(Qt.Orientation.Horizontal, self)
        self.positionSlider.sliderMoved.connect(self.setPosition)
        controlsLayout.addWidget(self.positionSlider)


        layout.addLayout(controlsLayout)

        # Connect media player signals
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.mediaStatusChanged.connect(self.mediaStateChanged)

        self.setLayout(layout)

    def nextVideo(self):
        """
        Avanza al siguiente video en la lista y guarda el JSON del video anterior.
        """
        if self.current_video_index:
            self.save_json_file()  # Guardar etiquetas del video anterior
        if self.video_files and self.current_video_index < len(self.video_files):
            next_video_path = self.video_files[self.current_video_index]
            self.load_and_play_video(next_video_path)
            self.current_video_index += 1


    def load_and_play_video(self, path):
        """
        Carga y reproduce un archivo de video/audio.
        
        Args:
            path: Ruta al archivo multimedia
        """
        self.current_media_path = path  # Almacenar ruta del archivo actual
        self.mediaPlayer.setSource(QUrl.fromLocalFile(path))
        self.videoFileSelected.emit(path)



    def save_json_file(self):
        """
        Guarda las etiquetas del video actual en un archivo JSON.
        Incluye clases, longitud del video y las pulsaciones de botones.
        """

        data = {
        #'project_name': self.second_window.project_name,
        #'labeler': self.second_window.labeler,
        #'method': self.second_window.method,
        'classes': self.second_window.classes,
        #'decision_basis': self.second_window.decision_basis,
        'video_lenght_frame': self.total_frames, # Include the frame lenght of a video
        'button_presses': self.second_window.buttonsWidget.get_formatted_labels(), # Assuming this method exists
        'button_presses_ms': self.second_window.buttonsWidget.get_formatted_labels(True)
        }
        if self.current_video_index != self.video_counter:
            data['button_presses'] = ''
            data['button_presses_ms'] = ''

        #input_file_name = "labeled" # Extract filename without extension
        filename = f"{self.video_names[self.current_video_index - 1]}.json"

        #options = QFileDialog.Option.ReadOnly
        filepath = f"{self.save_location_folder}\\{filename}"

        if filepath:
            # Save the JSON data
            with open(filepath, 'w') as file:
                json.dump(data, file, indent=4)

            QMessageBox.information(self, "File Saved", "Data saved successfully!")




    def openMediaFile(self):
        if hasattr(self, 'load_video_btn'):
            file_filter = "Video Files (*.mp4 *.flv *.ts *.mts *.avi)"
        elif hasattr(self, 'load_audio_btn'):
            file_filter = "Audio Files (*.mp3 *.wav)"
        elif hasattr(self, 'load_folder_btn'):
            pass
        else:
            return

        if hasattr(self, 'load_folder_btn'):
            folder = QFileDialog.getExistingDirectory(self, "Select Video Folder", ".", QFileDialog.Option.ShowDirsOnly)
            self.save_location_folder = QFileDialog.getExistingDirectory(self, "Select a folder to save data", ".", QFileDialog.Option.ShowDirsOnly)
            list_of_extensions = ['.mp4', '.flv', '.ts', '.mts', '.avi']
            video_files = []
            print(folder)
            for i in os.listdir(folder):
                if any(i.endswith(ext) for ext in list_of_extensions):
                    self.video_names.append(i.split('.'))
                    video_file_path = os.path.join(folder, i)
                    video_files.append(video_file_path)
            self.video_files = video_files
            self.nextVideo()


        else :
            fileName, _ = QFileDialog.getOpenFileName(self, "Select Media", ".",
            file_filter)
            if fileName != '':
                self.load_and_play_video(fileName)
                # Hide the select button and show the play button, speed button,
                # and progression slider
                if hasattr(self, 'load_video_btn'):
                    self.load_video_btn.hide()
                elif hasattr(self, 'load_audio_btn'):
                    self.load_audio_btn.hide()


    def play(self):
        if self.mediaPlayer.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        else:
            self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        video_capture = cv2.VideoCapture(self.video_files[self.current_video_index - 1])
        self.total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.positionSlider.setRange(0, duration)
        video_capture.release()

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def changePlaybackSpeed(self):
        """Cambia la velocidad de reproducción entre 1.0x, 0.5x y 0.10x"""
        speed = self.mediaPlayer.playbackRate()
        if speed == 1.0:
            self.mediaPlayer.setPlaybackRate(0.5)
            self.speedButton.setText('0.5x')
        elif speed == 0.5:
            self.mediaPlayer.setPlaybackRate(0.10)
            self.speedButton.setText('0.10x')
        else:
            self.mediaPlayer.setPlaybackRate(1.0)
            self.speedButton.setText('1.0x')

    def current_position_msec(self):
        """Retorna la posición actual del reproductor en milisegundos"""
        return self.mediaPlayer.position()

    def total_frames_transform(self):
        """Convierte la duración total de milisegundos a frames (asumiendo 60 fps)"""
        total_duration_msec = self.duration
        return int(round(total_duration_msec / 1000 * 60, 0))

class TimeBeamWidget(QWidget):
    """
    Widget que muestra una línea de tiempo con marcas de las etiquetas realizadas.
    Visualiza cuándo se ha etiquetado cada clase en el video.
    """
    def __init__(self, total_frames, mediaplayer, parent=None):
        """
        Inicializa el widget de línea de tiempo.
        
        Args:
            total_frames: Número total de frames del video
            mediaplayer: Referencia al reproductor de medios
            parent: Widget padre (opcional)
        """
        super().__init__(parent)
        self.total_frames = total_frames  # Total de frames del video
        self.marks = []  # Lista de marcas (frame, color) en la línea de tiempo
        self.setFixedHeight(50)
        self.mediaplayer = mediaplayer
        mediaplayer.mediaPlayer.durationChanged.connect(mediaplayer.total_frames_transform)

    def add_mark(self, frame_number, color):
        """
        Añade una marca en la línea de tiempo en el frame especificado.
        
        Args:
            frame_number: Número de frame donde añadir la marca
            color: Color de la marca
        """
        if self.mediaplayer.video_counter != self.mediaplayer.current_video_index:
            self.marks = []  # Limpiar marcas si cambió de video
            self.mediaplayer.video_counter = self.mediaplayer.current_video_index
        self.marks.append((frame_number, color))
        self.update()  # Redibujar el widget

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(QColor(0, 0, 0))) # Black color

        # Draw the time beam line
        painter.drawLine(10, self.height() // 2, self.width() - 10, self.height
        () // 2)

        # Fill the background with a light gray color to see the widget
        painter.fillRect(self.rect(), QColor(220, 220, 220))
        for index, (frame_number, color) in enumerate(self.marks):
            x_position = int((frame_number / self.mediaplayer.total_frames) *
            self.width())

            # Ensure the color is a QColor object
            if isinstance(color, str):
                color = QColor(color)

            # Draw colored line
            painter.setPen(QPen(color))
            painter.drawLine(x_position, self.height() // 2 - 10, x_position,
            self.height() // 2 + 10)

            y_text_position = self.height() // 2 + 25

            # Draw frame number in black
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(x_position - 10, y_text_position, str(frame_number)
            )

    def remove_marks_after_frame(self, frame_number):
        self.marks = [(frame, color) for frame, color in self.marks if frame <=
        frame_number]
        self.update()

class InformationWidget(QWidget):
    """
    Widget que muestra la información del proyecto de etiquetado.
    Incluye nombre, etiquetador, método, clases y base de decisión.
    """
    def __init__(self, project_name, labeler, method, classes, decision_basis,
        sequence_width, parent=None):
        """
        Inicializa el widget de información.
        
        Args:
            project_name: Nombre del proyecto
            labeler: Nombre de la persona que etiqueta
            method: Método de etiquetado ('One-hot-encoding' o 'Sequences')
            classes: Lista de clases a etiquetar
            decision_basis: Base de decisión ('Video', 'Audio', 'Intuition', 'Folder')
            sequence_width: Ancho de secuencia (si aplica)
            parent: Widget padre (opcional)
        """
        super().__init__(parent)

        layout = QVBoxLayout(self)

        # Display information
        label_project = QLabel(project_name)
        label_project.setFont(QFont("Arial", 24))
        layout.addWidget(label_project, alignment=Qt.AlignmentFlag.AlignCenter)

        label_labeler = QLabel(f"Labeler: {labeler}")
        label_labeler.setFont(QFont("Arial", 12))
        layout.addWidget(label_labeler, alignment=Qt.AlignmentFlag.AlignCenter)

        label_method = QLabel(f"Method: {method}")
        label_method.setFont(QFont("Arial", 12))
        layout.addWidget(label_method, alignment=Qt.AlignmentFlag.AlignCenter)

        # Display sequence width if method is "Sequences" and sequence_width is
        # provided
        if method == "Sequences" and sequence_width:
            label_sequence_width = QLabel(f"Sequence Width: {sequence_width}")
            label_sequence_width.setFont(QFont("Arial", 12))
            layout.addWidget(label_sequence_width, alignment=Qt.AlignmentFlag.AlignCenter)

        classes_text = ', '.join(classes)
        label_classes = QLabel(f"Classes: {classes_text}")
        label_classes.setFont(QFont("Arial", 12))
        layout.addWidget(label_classes, alignment=Qt.AlignmentFlag.AlignCenter)

        label_basis = QLabel(f"Basis of Decision: {decision_basis}")
        label_basis.setFont(QFont("Arial", 12))
        layout.addWidget(label_basis, alignment=Qt.AlignmentFlag.AlignCenter)

        # Separator line
        separator = QFrame(self)
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        self.setLayout(layout)

class ClassButtonsWidget(QWidget):
    """
    Widget que contiene los botones de clase para etiquetar.
    Cada botón representa una clase y tiene un atajo de teclado asociado.
    """
    def __init__(self, classes, media_player_widget, time_beam_widget, parent=
        None):
        """
        Inicializa los botones de clases.
        
        Args:
            classes: Lista de nombres de clases
            media_player_widget: Referencia al reproductor de medios
            time_beam_widget: Referencia al widget de línea de tiempo
            parent: Widget padre (opcional)
        """
        super().__init__(parent)
        self.media_player_widget = media_player_widget
        self.timeBeamWidget = time_beam_widget

        layout = QVBoxLayout(self)

        # Listas para almacenar las etiquetas realizadas
        self.labeled_frames = []  # Etiquetas con número de frame
        self.labeled_frames_ms = []  # Etiquetas con tiempo en milisegundos

        # Generar colores únicos para cada botón/clase
        self.colors = self.generate_colors(len(classes))

        # Define key sequences for shortcuts (you can extend or modify this list
        # )
        key_sequences = [f"Ctrl+{i}" for i in range(1, len(classes) + 1)]

        # Add a button for each class and associate it with a shortcut
        for i, (class_name, key_seq) in enumerate(zip(classes, key_sequences)):
            btn = QPushButton(f"{class_name} ({key_seq})", self)
            btn.setStyleSheet(f"background-color: {self.colors[i]}") # Set the
            # button’s background color
            btn.clicked.connect(self.buttonClicked) # Connect to slot

            # Create and associate a shortcut with the button
            shortcut = QShortcut(QKeySequence(key_seq), self)
            shortcut.activated.connect(btn.click)

            layout.addWidget(btn)

        self.setLayout(layout)

    @pyqtSlot()
    def buttonClicked(self):
        """
        Manejador del evento de clic en un botón de clase.
        Registra la etiqueta en el frame actual del video.
        """
        sender = self.sender()
        if sender:
            # Obtener nombre de la clase (sin el texto del atajo)
            class_name = sender.text().split(" (")
            
            # Encontrar índice del botón para obtener su color
            btn_index = [btn for btn in self.children() if isinstance(btn,
            QPushButton)].index(sender)
            btn_color = self.colors[btn_index]

            # Obtener posición actual en milisegundos
            msec_position = self.media_player_widget.current_position_msec()
            
            # Convertir milisegundos a número de frame (asumiendo 60 fps)
            frame_number = int(round(msec_position / 1000 * 60, 0))

            # Limpiar etiquetas si cambió de video
            if self.media_player_widget.current_video_index != self.media_player_widget.video_counter:
                self.labeled_frames = []
                self.labeled_frames_ms = []

            # Añadir la etiqueta actual a la lista
            self.labeled_frames.append((class_name, frame_number))
            self.labeled_frames_ms.append((class_name, msec_position))

            # Añadir marca visual en la línea de tiempo
            self.timeBeamWidget.add_mark(frame_number, btn_color)


    def handle_video_frame(self):
        self.current_frame = 0


    def generate_colors(self, num_colors):
        """
        Genera una lista de colores únicos para los botones.
        
        Args:
            num_colors: Número de colores a generar
            
        Returns:
            Lista de colores en formato hexadecimal
        """
        colors = []
        step = 360 / num_colors  # Dividir el círculo de matiz en partes iguales
        for i in range(num_colors):
            hue = int(i * step)  # Calcular matiz para color actual
            colors.append(QColor.fromHsv(hue, 255, 255).name())
        return colors

    def get_formatted_labels(self, ms = False):
        """
        Formatea las etiquetas realizadas en una cadena de texto.
        
        Args:
            ms: Si es True, usa milisegundos; si es False, usa frames
            
        Returns:
            Cadena con formato "Clase: Frame; Clase: Frame; ..."
        """
        formatted_labels = []
        if ms :
            for label, frame in self.labeled_frames_ms:
                formatted_labels.append(f"{label}: {frame}")
        else:
            for label, frame in self.labeled_frames:
                formatted_labels.append(f"{label}: {frame}")
        return '; '.join(formatted_labels)

class SecondWindow(QMainWindow):
    """
    Ventana principal de etiquetado donde se reproduce el video/audio
    y se registran las etiquetas mediante los botones de clase.
    """
    def __init__(self, project_name, labeler, method, classes, decision_basis,
        sequence_width):
        """
        Inicializa la ventana de etiquetado.
        
        Args:
            project_name: Nombre del proyecto
            labeler: Nombre del etiquetador
            method: Método de etiquetado
            classes: Lista de clases
            decision_basis: Base de decisión ('Video', 'Audio', 'Folder')
            sequence_width: Ancho de secuencia
        """
        super().__init__()

        self.setWindowTitle("Labeling Project Overview")

        self.resize(400,200)

        layout = QVBoxLayout()

        # Guardar los valores pasados como variables de instancia
        self.project_name = project_name
        self.labeler = labeler
        self.method = method
        self.classes = classes
        self.decision_basis = decision_basis
        self.sequence_width = sequence_width
        self.total_frames = 0  # Total de frames del video actual

        # Info widget setup
        self.infoWidget = InformationWidget(project_name, labeler, method,
        classes, decision_basis, sequence_width)
        layout.addWidget(self.infoWidget)

        # Media player widget setup
        self.mediaWidget = MediaPlayerWidget(decision_basis, self)
        layout.addWidget(self.mediaWidget)

        # Connect the videoFileSelected signal after the mediaWidget is created
        self.mediaWidget.videoFileSelected.connect(self.on_video_file_selected)

        # Time beam widget setup (initially set total frames to 0)
        self.timeBeamWidget = TimeBeamWidget(0,self.mediaWidget)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.addWidget(self.timeBeamWidget)


        # Class buttons widget setup
        self.buttonsWidget = ClassButtonsWidget(classes, self.mediaWidget, self.timeBeamWidget)
        layout.addWidget(self.buttonsWidget)

        # Setting the main layout
        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def on_video_file_selected(self, video_path):
        """
        Callback cuando se selecciona un archivo de video.
        Actualiza el total de frames y la línea de tiempo.
        
        Args:
            video_path: Ruta al archivo de video seleccionado
        """
        self.total_frames = self.mediaWidget.total_frames_transform()
        self.timeBeamWidget.total_frames = self.total_frames
        self.timeBeamWidget.update()



    def closeEvent(self, event):
        """
        Manejador del evento de cierre de ventana.
        Guarda las etiquetas en un archivo JSON antes de cerrar.
        """
        print("Closing the window!")

        # Recuperar datos de la información y botones
        data = {
        #'project_name': self.project_name,
        #'labeler': self.labeler,
        #'method': self.method,
        'classes': self.classes,
        #'decision_basis': self.decision_basis,
        'sequence_width': self.sequence_width, # Include the sequence width
        'button_presses': self.buttonsWidget.get_formatted_labels(), # Assuming
        # this method exists
        'button_presses_ms': self.buttonsWidget.get_formatted_labels(True)
        }

        # JSON filename formatting
        input_file_name = "labeled" # Extract filename without extension
        filename = f"{input_file_name}_{self.method}_labels.json"

        # Open filedialog for user to select save location
        options = QFileDialog.Option.ReadOnly
        filepath, _ = QFileDialog.getSaveFileName(self, "Save File", filename, "JSON Files (*.json);;All Files (*)", options=options)


        if filepath:
            # Save the JSON data
            with open(filepath, 'w') as file:
                json.dump(data, file, indent=4)

            QMessageBox.information(self, "File Saved", "Data saved successfully!")

        event.accept()


if __name__ == "__main__":
    app = 0
    app = QApplication(sys.argv)
    win = FirstWindow()
    win.show()
    sys.exit(app.exec())