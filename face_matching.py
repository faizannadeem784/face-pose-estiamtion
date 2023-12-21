import cv2
from facePose import *

def compare_faces(face1, face2):
    """
    Compare two faces using histogram correlation.

    Args:
        face1: The first face image.
        face2: The second face image.

    Returns:
        correlation: The correlation coefficient between the histograms of the faces.
    """
    # Convert faces to grayscale
    gray_face1 = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
    gray_face2 = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)

    # Resize faces to the same dimensions
    resized_face1 = cv2.resize(gray_face1, (100, 100))
    resized_face2 = cv2.resize(gray_face2, (100, 100))

    # Calculate histogram for each face
    hist_face1 = cv2.calcHist([resized_face1], [0], None, [256], [0, 256])
    hist_face2 = cv2.calcHist([resized_face2], [0], None, [256], [0, 256])

    # Normalize histograms
    cv2.normalize(hist_face1, hist_face1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_face2, hist_face2, 0, 1, cv2.NORM_MINMAX)

    # Calculate the correlation coefficient between the histograms
    correlation = cv2.compareHist(hist_face1, hist_face2, cv2.HISTCMP_CORREL)

    return correlation

def match_face_in_video_with_picture(video_path, picture_path, threshold=0.5):
    """
    Match a face in a video with a picture using histogram correlation.

    Args:
        video_path: The path to the video file.
        picture_path: The path to the picture file.
        threshold: The correlation threshold for considering a match.

    Returns:
        is_match: True if a match is found, False otherwise.
    """
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the picture
    picture = cv2.imread(picture_path)

    # Detect faces in the picture
    faces_picture = face_cascade.detectMultiScale(picture, scaleFactor=1.3, minNeighbors=5)

    # Check if there is at least one face in the picture
    if len(faces_picture) > 0:
        # Get the first face in the picture
        x_picture, y_picture, w_picture, h_picture = faces_picture[0]
        face_picture = picture[y_picture:y_picture + h_picture, x_picture:x_picture + w_picture]

        # Read the video
        cap = cv2.VideoCapture(video_path)

        is_match = False  # Initialize the match variable

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces_frame = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

            # Check if there is at least one face in the frame
            if len(faces_frame) > 0:
                # Get the first face in the frame
                x_frame, y_frame, w_frame, h_frame = faces_frame[0]
                face_frame = frame[y_frame:y_frame + h_frame, x_frame:x_frame + w_frame]

                # Compare the faces using histogram correlation
                correlation = compare_faces(face_picture, face_frame)

                # Compare the correlation with the threshold
                if correlation > threshold:
                    is_match = True
                    print(f"Face Match! (Correlation: {correlation:.2f})")
                    break
                else:
                    is_match = False
                    print(f"Face Not Match! (Correlation: {correlation:.2f})")

            # Display the frame with rectangles around the detected faces
            for (x, y, w, h) in faces_frame:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow("Video", frame)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture object and close all windows
        cap.release()
        cv2.destroyAllWindows()

        print(f"Final Result: {is_match}")
        return is_match
    else:
        text = "No faces detected in the picture."
        print(text)
        return False  # No faces in the picture, so it's not a match


if __name__ == "__main__":
    video_path = 'viedo.mp4'
    picture_path = 'idcard.jpg'
    is_face_match = match_face_in_video_with_picture(video_path, picture_path)
        # Check the result
    if is_face_match:
        print("your Face is match!")
        main()
    else:
        print("No face match.")
