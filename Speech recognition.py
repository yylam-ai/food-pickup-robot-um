#!/usr/bin/env python

import rospy
import speech_recognition as sr
from gtts import gTTS
import os
from std_msgs.msg import String

class SpeechProcessingNode:
    def __init__(self):
        rospy.init_node('speech_processing_node')
        self.pub = rospy.Publisher('speech_to_text', String, queue_size=10)
        rospy.Subscriber('text_to_speech', String, self.tts_callback)
        rospy.loginfo("Speech Processing Node Initialized")
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Start speech recognition
        self.listen_to_speech()

    def listen_to_speech(self):
        while not rospy.is_shutdown():
            try:
                rospy.loginfo("Listening for speech...")
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source)
                    audio = self.recognizer.listen(source)

                # Recognize speech
                text = self.recognizer.recognize_google(audio)
                rospy.loginfo(f"Recognized text: {text}")
                self.pub.publish(text)

            except sr.UnknownValueError:
                rospy.logwarn("Could not understand the audio.")
            except sr.RequestError as e:
                rospy.logerr(f"Speech Recognition service error: {e}")
            except rospy.ROSInterruptException:
                break

    def tts_callback(self, msg):
        text = msg.data
        rospy.loginfo(f"Converting text to speech: {text}")
        self.text_to_speech(text)

    def text_to_speech(self, text):
        try:
            tts = gTTS(text=text, lang='en')
            tts.save("/tmp/tts_output.mp3")
            os.system("mpg123 /tmp/tts_output.mp3")
            rospy.loginfo("Text-to-Speech completed.")
        except Exception as e:
            rospy.logerr(f"Failed to convert text to speech: {e}")

if __name__ == "__main__":
    try:
        SpeechProcessingNode()
    except rospy.ROSInterruptException:
        pass
