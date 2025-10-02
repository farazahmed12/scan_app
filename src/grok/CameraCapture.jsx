import * as handpose from '@tensorflow-models/handpose';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl'; // GPU acceleration
import React, { useEffect, useRef, useState } from 'react';

function CameraCapture() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [fingerCount, setFingerCount] = useState(0);
  const [distanceMessage, setDistanceMessage] = useState('');
  const [capturedImage, setCapturedImage] = useState(null);
  const modelRef = useRef(null);
  const runningRef = useRef(false);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Resize to full screen
    const setFullScreen = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      video.width = width;
      video.height = height;
      canvas.width = width;
      canvas.height = height;
    };
    setFullScreen();
    window.addEventListener('resize', setFullScreen);

    // Get camera access
    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } })
      .then(stream => {
        video.srcObject = stream;
        video.play();
        console.log('Camera access granted');
      })
      .catch(err => console.error('Camera access error:', err));

    // Load Handpose model
    const loadModel = async () => {
      try {
        await tf.setBackend('webgl');
        modelRef.current = await handpose.load();
        console.log('Handpose model loaded');
        runningRef.current = true;
        predictWebcam();
      } catch (error) {
        console.error('Error loading Handpose model:', error);
      }
    };

    loadModel();

    // Prediction loop
    const predictWebcam = async () => {
      if (!runningRef.current || !modelRef.current || !video || video.readyState !== 4) {
        requestAnimationFrame(predictWebcam);
        return;
      }

      const predictions = await modelRef.current.estimateHands(video);
      processResults(predictions, ctx, canvas);

      requestAnimationFrame(predictWebcam);
    };

    return () => {
      window.removeEventListener('resize', setFullScreen);
      runningRef.current = false;
    };
  }, []);

  const processResults = (predictions, ctx, canvas) => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Mirror effect
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);

    // Draw video feed
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    ctx.restore();

    let detectedFingers = 0;
    let newDistanceMessage = '';

    if (predictions && predictions.length > 0) {
      const landmarks = predictions[0].landmarks;

      // ✅ Finger open/close detection
      const isFingerOpen = (tip, pip) => landmarks[tip][1] < landmarks[pip][1]; // y-axis check
      const thumbOpen = landmarks[4][0] > landmarks[3][0]; // for right hand (mirrored)
      const indexOpen = isFingerOpen(8, 6);
      const middleOpen = isFingerOpen(12, 10);
      const ringOpen = isFingerOpen(16, 14);
      const pinkyOpen = isFingerOpen(20, 18);

      detectedFingers = [thumbOpen, indexOpen, middleOpen, ringOpen, pinkyOpen]
        .filter(Boolean).length;

      // ✅ Distance estimation
      const wrist = landmarks[0];
      const middleTip = landmarks[12];
      if (wrist && middleTip) {
        const dx = middleTip[0] - wrist[0];
        const dy = middleTip[1] - wrist[1];
        const handHeight = Math.sqrt(dx ** 2 + dy ** 2);
        const normalizedHeight = handHeight / canvas.height;

        if (normalizedHeight < 0.3) {
          newDistanceMessage = 'Too Far - Move Closer';
        } else if (normalizedHeight > 0.7) {
          newDistanceMessage = 'Too Close - Move Back';
        } else {
          newDistanceMessage = 'Perfect Distance';
        }
      }

      // ✅ Auto-capture when 5 fingers + perfect distance
      if (detectedFingers === 5 && newDistanceMessage === 'Perfect Distance' && !capturedImage) {
        const captureCanvas = document.createElement('canvas');
        captureCanvas.width = videoRef.current.videoWidth;
        captureCanvas.height = videoRef.current.videoHeight;
        captureCanvas.getContext('2d').drawImage(videoRef.current, 0, 0);
        const imageData = captureCanvas.toDataURL('image/jpeg');
        setCapturedImage(imageData);
        console.log('Captured Image:', imageData);
      }
    }

    setFingerCount(detectedFingers);
    setDistanceMessage(newDistanceMessage);

    // Overlay UI
    ctx.font = '32px Arial';
    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 3;
    ctx.strokeText(`Fingers: ${detectedFingers}/5`, 30, 50);
    ctx.fillText(`Fingers: ${detectedFingers}/5`, 30, 50);
    ctx.strokeText(distanceMessage, 30, 100);
    ctx.fillText(distanceMessage, 30, 100);
  };

  return (
    <div className="camera-container">
      <video ref={videoRef} style={{ display: 'none' }} playsInline />
      <canvas ref={canvasRef} className="camera-canvas" />
      {capturedImage && (
        <div className="capture-preview">
          <h3>Captured!</h3>
          <img src={capturedImage} alt="Captured Hand" />
        </div>
      )}
    </div>
  );
}

export default CameraCapture;
