import * as handPoseDetection from "@tensorflow-models/hand-pose-detection";
import "@tensorflow/tfjs-backend-webgl";
import * as tf from "@tensorflow/tfjs-core";
import React, { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";

const HandCapture = () => {
  const webcamRef = useRef(null);
  const [detector, setDetector] = useState(null);
  const [capturedImage, setCapturedImage] = useState(null);

  useEffect(() => {
    const loadModel = async () => {
      await tf.setBackend("webgl");
      await tf.ready();

      const model = handPoseDetection.SupportedModels.MediaPipeHands;
      const detectorConfig = {
        runtime: "mediapipe",
        solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/hands", // ✅ use CDN
        modelType: "lite",
      };

      const detectorInstance = await handPoseDetection.createDetector(
        model,
        detectorConfig
      );
      setDetector(detectorInstance);
    };

    loadModel();
  }, []);

  useEffect(() => {
    if (!detector) return;
    const interval = setInterval(() => detectHands(), 300);
    return () => clearInterval(interval);
  }, [detector]);

  const detectHands = async () => {
    if (
      webcamRef.current &&
      webcamRef.current.video &&
      webcamRef.current.video.readyState === 4
    ) {
      const video = webcamRef.current.video;
      const hands = await detector.estimateHands(video);

      if (hands.length > 0) {
        const hand = hands[0];
        const fingers = hand.keypoints.filter((k) =>
          k.name.includes("tip")
        );

        if (fingers.length === 5) {
          capturePhoto(video);
        }
      }
    }
  };

  const capturePhoto = (video) => {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const image = canvas.toDataURL("image/png");
    setCapturedImage(image);
  };

  return (
    <div>
      <Webcam
        ref={webcamRef}
        style={{ width: 640, height: 480, borderRadius: "10px" }}
      />
      {capturedImage && (
        <div>
          <h3>Captured Image:</h3>
          <img src={capturedImage} alt="captured" width={320} />
        </div>
      )}
    </div>
  );
};

export default HandCapture;
