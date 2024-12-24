import cv2
from fer import FER
import random
import time
import tkinter as tk
from tkinter import filedialog, Label, Toplevel
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

detector = FER()

# A simple dictionary for song recommendations based on emotions (Arijit Singh's songs)
song_recommendations = {
    "happy": [
        "Tum Hi Ho - Arijit Singh",
        "Raabta - Arijit Singh",
        "Chal Ghar Chalen - Arijit Singh",
        "Tum Jo Aaye - Arijit Singh"
    ],
    "sad": [
        "Channa Mereya - Arijit Singh",
        "Agar Tum Saath Ho - Arijit Singh",
        "Muskurane - Arijit Singh",
        "Phir Bhi Tumko Chaahunga - Arijit Singh"
    ],
    "angry": [
        "Dilliwaali Girlfriend - Arijit Singh",
        "Malang - Arijit Singh",
        "Khuda Jaane - Arijit Singh",
        "Pardesi - Arijit Singh"
    ],
    "surprised": [
        "Badtameez Dil - Arijit Singh",
        "Dil Dhadakne Do - Arijit Singh",
        "Ae Dil Hai Mushkil - Arijit Singh",
        "Ae Mere Humsafar - Arijit Singh"
    ],
    "neutral": [
        "Raabta - Arijit Singh",
        "Agar Tum Saath Ho - Arijit Singh",
        "Tum Mile - Arijit Singh",
        "Tum Tak - Arijit Singh"
    ]
}

def recommend_song(emotion):
    """Recommend a sqqqong based on the detected emotion."""
    if emotion in song_recommendations:
        return random.choice(song_recommendations[emotion])
    return "No recommendations available."


def show_recommendations(emotion):
    """Show song recommendations in a new GUI window."""
    recommendations = song_recommendations.get(emotion, [])

    # Create a new top-level window
    recommendations_window = Toplevel()
    recommendations_window.title(f"Recommendations for {emotion.capitalize()}")

    # Add labels for each recommended song
    Label(recommendations_window, text=f"Recommended Songs for '{emotion}':", font=('Helvetica', 14)).pack(pady=10)

    for song in recommendations:
        Label(recommendations_window, text=song, font=('Helvetica', 12)).pack(anchor='w')

    # Button to close the recommendations window
    close_button = tk.Button(recommendations_window, text="Close", command=recommendations_window.destroy)
    close_button.pack(pady=10)


def plot_emotion_graph(emotions):
    """Plot emotion probabilities on a graph."""
    emotion_names = list(emotions.keys())
    emotion_values = list(emotions.values())

    plt.clf()  # Clear the figure to prevent overlapping graphs
    plt.bar(emotion_names, emotion_values, color='royalblue')
    plt.title("Emotion Probabilities")
    plt.xlabel("Emotion")
    plt.ylabel("Probability")
    plt.ylim(0, 1)  # Set the y-axis range to 0-1 (probability range)

    plt.draw()
    plt.pause(0.01)  # Pause for a short time to update the graph


def detect_emotion_from_image(image_path):
    """Detect emotion from a given image."""
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read the image from {image_path}")
        return

    # Detect emotions in the image
    emotions = detector.detect_emotions(image)

    # Draw bounding boxes and labels on the detected faces
    for emotion in emotions:
        (x, y, w, h) = emotion['box']
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get the emotion with the highest score
        emotion_label = emotion['emotions']
        max_emotion = max(emotion_label, key=emotion_label.get)
        cv2.putText(image, max_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Recommend a song based on the detected emotion
        show_recommendations(max_emotion)

        # Store emotion probabilities for the graph
        emotion_probabilities = emotion_label

    # Plot the emotion probabilities graph
    plot_emotion_graph(emotion_probabilities)

    # Display the result
    cv2.imshow("Emotion Detection - Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_emotion_from_camera():
    """Detect emotion from webcam input."""
    # Start capturing video from the webcam
    video_capture = cv2.VideoCapture(0)
    start_time = time.time()  # Start the timer

    # Create a matplotlib figure for the emotion graph
    plt.figure("Emotion Graph", figsize=(5, 5))
    plt.ion()  # Turn on interactive mode for live plotting

    while True:
        # Read a frame from the video capture
        ret, frame = video_capture.read()
        if not ret:
            break

        # Initialize emotion_probabilities
        emotion_probabilities = {}  # Initialize to avoid UnboundLocalError

        # Detect emotions in the frame
        emotions = detector.detect_emotions(frame)

        # Draw bounding boxes and labels on the detected faces
        if emotions:  # Check if any emotions were detected
            for emotion in emotions:
                (x, y, w, h) = emotion['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Get the emotion with the highest score
                emotion_label = emotion['emotions']
                max_emotion = max(emotion_label, key=emotion_label.get)
                cv2.putText(frame, max_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                # Show recommendations for the detected emotion
                show_recommendations(max_emotion)

                # Store emotion probabilities for the graph
                emotion_probabilities = emotion_label

            # Plot the emotion probabilities graph
            plot_emotion_graph(emotion_probabilities)

        # Display the frame with the detected emotions and recommendations
        cv2.imshow("Emotion Detection - Camera", frame)

        # Break the loop on 'q' key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit when 'q' is pressed
            break

    # Release the video capture and close windows
    video_capture.release()
    cv2.destroyAllWindows()
    plt.ioff()  # Turn off interactive mode to stop live plotting
    plt.show()  # Display the final plot


# GUI for the application
def browse_image():
    """Browse an image from the file system and detect emotion."""
    image_path = filedialog.askopenfilename()
    if image_path:
        detect_emotion_from_image(image_path)


def start_camera():
    """Start emotion detection using the webcam."""
    detect_emotion_from_camera()


# Main GUI window setup
root = tk.Tk()
root.title("Emotion Detection and Song Recommendation")
root.geometry("400x400")

# Add buttons to choose webcam or image
btn_image = tk.Button(root, text="Detect Emotion from Image", command=browse_image, height=2, width=30)
btn_image.pack(pady=20)

btn_camera = tk.Button(root, text="Detect Emotion from Camera", command=start_camera, height=2, width=30)
btn_camera.pack(pady=20)

# Start the GUI loop
root.mainloop()
