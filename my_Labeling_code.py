# Listing A.1: Labeling code
# Script to label football (soccer) video/audio via graphical interface
# WAS NOT INCLUDED IN THE CODE
# created from the PDF file.

# Import necessary libraries for the graphical interface
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QLineEdit,
QPushButton, QComboBox, QVBoxLayout, QWidget, QFileDialog, QFrame,
QMessageBox, QSlider, QHBoxLayout, QStyle)
from PyQt6.QtGui import *
from PyQt6.QtCore import QStandardPaths, Qt, QUrl, pyqtSlot, pyqtSignal
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput, QVideoSink
from PyQt6.QtMultimediaWidgets import QVideoWidget
import cv2  # OpenCV for video processing
import time
import os

import json  # To save and load configurations
import sys

class FirstWindow(QMainWindow):
    """
    Main configuration window for the labeling project.
    Allows configuring the project name, labeling method, classes, and decision basis.
    """
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """Initializes the user interface with all configuration fields"""
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

        # Sequence width add
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
        Opens the secondary labeling window with the current configuration.
        Gathers all parameters and passes them to SecondWindow.
        """
        # Gather values from input fields
        project_name = self.entry_name.text()
        labeler = self.entry_person.text()
        method = self.dropdown_method.currentText()
        classes = self.entry_classes.text().split(',') # assuming comma-separated
        decision_basis = self.dropdown_basis.currentText()
        sequence_width = self.entry_sequence_width.text()

        # Create and show the labeling window
        self.second_window = SecondWindow(project_name, labeler, method, classes, decision_basis, sequence_width)
        self.second_window.show()
        self.close()

    def load_settings(self):
        """Loads configuration from a JSON file"""
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
                self.on_method_changed(method_index) # Trigger the visibility change based on loaded method
            self.entry_classes.setText(data.get('classes', ''))
            basis_index = self.dropdown_basis.findText(data.get('decision_basis', ''))
            if basis_index != -1:
                self.dropdown_basis.setCurrentIndex(basis_index)
            self.entry_sequence_width.setText(data.get('sequence_width', ''))
            # Populate the sequence width field

            QMessageBox.information(self, "Load Settings", "Settings loaded successfully!")

    def save_settings(self):
        """Saves current configuration to a JSON file"""
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Settings",
        QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation), "JSON files (*.json);;All files (*)")

        if filepath:
            # Create dictionary with current configuration
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
        Callback when the labeling method changes.
        Shows/hides the sequence width field depending on the selected method.
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
    Media player widget (video/audio) for labeling.
    Allows playing individual files or processing entire folders.
    """
    videoFileSelected = pyqtSignal(str)  # Signal emitted when a video is selected

    def __init__(self, decision_basis, second_window, parent=None):
        """
        Initializes the media player.
        
        Args:
            decision_basis: Media type ('Video', 'Audio', or 'Folder')
            second_window: Reference to the main labeling window
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        # Player control variables
        self.duration = 0  # Total media duration
        self.mediaPlayer = QMediaPlayer(self)  # Multimedia player
        self.audioOutput = QAudioOutput(self)  # Audio output
        self.mediaPlayer.setAudioOutput(self.audioOutput)
        self.audioOutput.setVolume(100)
        
        # Variables for managing multiple videos
        self.video_files = []  # List of video files
        self.current_video_index = 0  # Current video index
        self.save_location_folder = ''  # Folder to save results
        self.video_names = []  # Video names
        self.second_window = second_window  # Reference to main window
        self.video_counter = self.current_video_index
        self.current_media_path = None  # Path of the current media file

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
        Advances to the next video in the list and saves the JSON of the previous video.
        """
        if self.current_video_index:
            self.save_json_file()  # Save labels of the previous video
        if self.video_files and self.current_video_index < len(self.video_files):
            next_video_path = self.video_files[self.current_video_index]
            self.load_and_play_video(next_video_path)
            self.current_video_index += 1

    def load_and_play_video(self, path):
        """
        Loads and plays a video/audio file.
        
        Args:
            path: Path to the multimedia file
        """
        self.current_media_path = path  # Store current file path
        self.mediaPlayer.setSource(QUrl.fromLocalFile(path))
        self.videoFileSelected.emit(path)

    def save_json_file(self):
        """
        Saves the current video labels to a JSON file.
        Includes classes, video length, and button presses.
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

        else:
            fileName, _ = QFileDialog.getOpenFileName(self, "Select Media", ".", file_filter)
            if fileName != '':
                self.load_and_play_video(fileName)
                # Hide the select button and show the play button, speed button, and progression slider
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
        """Changes the playback speed between 1.0x, 0.5x, and 0.10x"""
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
        """Returns the current player position in milliseconds"""
        return self.mediaPlayer.position()

    def total_frames_transform(self):
        """Converts total duration from milliseconds to frames (assuming 60 fps)"""
        total_duration_msec = self.duration
        return int(round(total_duration_msec / 1000 * 60, 0))

class TimeBeamWidget(QWidget):
    """
    Widget that displays a timeline with marks for the labels made.
    Visualizes when each class has been labeled in the video.
    """
    def __init__(self, total_frames, mediaplayer, parent=None):
        """
        Initializes the timeline widget.
        
        Args:
            total_frames: Total number of video frames
            mediaplayer: Reference to the media player
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        self.total_frames = total_frames  # Total video frames
        self.marks = []  # List of marks (frame, color) on the timeline
        self.setFixedHeight(50)
        self.mediaplayer = mediaplayer
        mediaplayer.mediaPlayer.durationChanged.connect(mediaplayer.total_frames_transform)

    def add_mark(self, frame_number, color):
        """
        Adds a mark on the timeline at the specified frame.
        
        Args:
            frame_number: Frame number where the mark is added
            color: Mark color
        """
        if self.mediaplayer.video_counter != self.mediaplayer.current_video_index:
            self.marks = []  # Clear marks if video changed
            self.mediaplayer.video_counter = self.mediaplayer.current_video_index
        self.marks.append((frame_number, color))
        self.update()  # Redraw the widget

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(QColor(0, 0, 0))) # Black color

        # Draw the time beam line
        painter.drawLine(10, self.height() // 2, self.width() - 10, self.height() // 2)

        # Fill the background with a light gray color to see the widget
        painter.fillRect(self.rect(), QColor(220, 220, 220))
        for index, (frame_number, color) in enumerate(self.marks):
            x_position = int((frame_number / self.mediaplayer.total_frames) * self.width())

            # Ensure the color is a QColor object
            if isinstance(color, str):
                color = QColor(color)

            # Draw colored line
            painter.setPen(QPen(color))
            painter.drawLine(x_position, self.height() // 2 - 10, x_position, self.height() // 2 + 10)

            y_text_position = self.height() // 2 + 25

            # Draw frame number in black
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(x_position - 10, y_text_position, str(frame_number))

    def remove_marks_after_frame(self, frame_number):
        self.marks = [(frame, color) for frame, color in self.marks if frame <= frame_number]
        self.update()

class InformationWidget(QWidget):
    """
    Widget displaying the labeling project information.
    Includes name, labeler, method, classes, and decision basis.
    """
    def __init__(self, project_name, labeler, method, classes, decision_basis, sequence_width, parent=None):
        """
        Initializes the information widget.
        
        Args:
            project_name: Project name
            labeler: Name of the person labeling
            method: Labeling method ('One-hot-encoding' or 'Sequences')
            classes: List of classes to label
            decision_basis: Decision basis ('Video', 'Audio', 'Intuition', 'Folder')
            sequence_width: Sequence width (if applicable)
            parent: Parent widget (optional)
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

        # Display sequence width if method is "Sequences" and sequence_width is provided
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
    Widget containing the class buttons for labeling.
    Each button represents a class and has an associated keyboard shortcut.
    """
    def __init__(self, classes, media_player_widget, time_beam_widget, parent=None):
        """
        Initializes the class buttons.
        
        Args:
            classes: List of class names
            media_player_widget: Reference to the media player
            time_beam_widget: Reference to the timeline widget
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        self.media_player_widget = media_player_widget
        self.timeBeamWidget = time_beam_widget

        layout = QVBoxLayout(self)

        # Lists to store the labels made
        self.labeled_frames = []  # Labels with frame number
        self.labeled_frames_ms = []  # Labels with time in milliseconds

        # Generate unique colors for each button/class
        self.colors = self.generate_colors(len(classes))

        # Define key sequences for shortcuts (you can extend or modify this list)
        key_sequences = [f"Ctrl+{i}" for i in range(1, len(classes) + 1)]

        # Add a button for each class and associate it with a shortcut
        for i, (class_name, key_seq) in enumerate(zip(classes, key_sequences)):
            btn = QPushButton(f"{class_name} ({key_seq})", self)
            btn.setStyleSheet(f"background-color: {self.colors[i]}") # Set the button’s background color
            btn.clicked.connect(self.buttonClicked) # Connect to slot

            # Create and associate a shortcut with the button
            shortcut = QShortcut(QKeySequence(key_seq), self)
            shortcut.activated.connect(btn.click)

            layout.addWidget(btn)

        self.setLayout(layout)

    @pyqtSlot()
    def buttonClicked(self):
        """
        Handler for the class button click event.
        Records the label at the current video frame.
        """
        sender = self.sender()
        if sender:
            # Get class name (without the shortcut text)
            class_name = sender.text().split(" (")
            
            # Find button index to get its color
            btn_index = [btn for btn in self.children() if isinstance(btn, QPushButton)].index(sender)
            btn_color = self.colors[btn_index]

            # Get current position in milliseconds
            msec_position = self.media_player_widget.current_position_msec()
            
            # Convert milliseconds to frame number (assuming 60 fps)
            frame_number = int(round(msec_position / 1000 * 60, 0))

            # Clear labels if video changed
            if self.media_player_widget.current_video_index != self.media_player_widget.video_counter:
                self.labeled_frames = []
                self.labeled_frames_ms = []

            # Add the current label to the list
            self.labeled_frames.append((class_name[0], frame_number))
            self.labeled_frames_ms.append((class_name[0], msec_position))

            # Add visual mark on the timeline
            self.timeBeamWidget.add_mark(frame_number, btn_color)

    def handle_video_frame(self):
        self.current_frame = 0

    def generate_colors(self, num_colors):
        """
        Generates a list of unique colors for the buttons.
        
        Args:
            num_colors: Number of colors to generate
            
        Returns:
            List of colors in hexadecimal format
        """
        colors = []
        step = 360 / num_colors  # Divide the hue circle into equal parts
        for i in range(num_colors):
            hue = int(i * step)  # Calculate hue for current color
            colors.append(QColor.fromHsv(hue, 255, 255).name())
        return colors

    def get_formatted_labels(self, ms=False):
        """
        Formats the made labels into a text string.
        
        Args:
            ms: If True, uses milliseconds; if False, uses frames
            
        Returns:
            String formatted as "Class: Frame; Class: Frame; ..."
        """
        formatted_labels = []
        if ms:
            for label, frame in self.labeled_frames_ms:
                formatted_labels.append(f"{label}: {frame}")
        else:
            for label, frame in self.labeled_frames:
                formatted_labels.append(f"{label}: {frame}")
        return '; '.join(formatted_labels)

class SecondWindow(QMainWindow):
    """
    Main labeling window where the video/audio is played
    and labels are recorded using the class buttons.
    """
    def __init__(self, project_name, labeler, method, classes, decision_basis, sequence_width):
        """
        Initializes the labeling window.
        
        Args:
            project_name: Project name
            labeler: Name of the labeler
            method: Labeling method
            classes: List of classes
            decision_basis: Decision basis ('Video', 'Audio', 'Folder')
            sequence_width: Sequence width
        """
        super().__init__()

        self.setWindowTitle("Labeling Project Overview")

        self.resize(400,200)

        layout = QVBoxLayout()

        # Save the passed values as instance variables
        self.project_name = project_name
        self.labeler = labeler
        self.method = method
        self.classes = classes
        self.decision_basis = decision_basis
        self.sequence_width = sequence_width
        self.total_frames = 0  # Total frames of the current video

        # Info widget setup
        self.infoWidget = InformationWidget(project_name, labeler, method, classes, decision_basis, sequence_width)
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
        Callback when a video file is selected.
        Updates the total frames and the timeline.
        
        Args:
            video_path: Path to the selected video file
        """
        self.total_frames = self.mediaWidget.total_frames_transform()
        self.timeBeamWidget.total_frames = self.total_frames
        self.timeBeamWidget.update()

    def closeEvent(self, event):
        """
        Window close event handler.
        Saves the labels to a JSON file before closing.
        """
        print("Closing the window!")

        # Retrieve data from info and buttons
        data = {
        #'project_name': self.project_name,
        #'labeler': self.labeler,
        #'method': self.method,
        'classes': self.classes,
        #'decision_basis': self.decision_basis,
        'sequence_width': self.sequence_width, # Include the sequence width
        'button_presses': self.buttonsWidget.get_formatted_labels(), # Assuming this method exists
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
