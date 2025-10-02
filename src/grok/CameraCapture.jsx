import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';
import { FilesetResolver, HAND_CONNECTIONS, HandLandmarker } from '@mediapipe/tasks-vision';
import React, { useEffect, useRef, useState } from 'react';

function CameraCapture() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [fingerCount, setFingerCount] = useState(0);
  const [distanceMessage, setDistanceMessage] = useState('');
  const [capturedImage, setCapturedImage] = useState(null);
  const handLandmarkerRef = useRef(null);
  const runningRef = useRef(false);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Full screen setup
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

    // Camera access
    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } })
      .then(stream => {
        video.srcObject = stream;
        video.play();
        console.log('Camera access granted');
      })
      .catch(err => console.error('Camera access error:', err));

    // Initialize HandLandmarker
    const initializeHandLandmarker = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm');
        handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: 'GPU'
          },
          runningMode: 'VIDEO',
          numHands: 1,
          minHandDetectionConfidence: 0.7,
          minHandPresenceConfidence: 0.7,
          minTrackingConfidence: 0.7,
        });
        console.log('HandLandmarker initialized');
        runningRef.current = true;
        predictWebcam();
      } catch (error) {
        console.error('Error initializing HandLandmarker:', error);
      }
    };

    initializeHandLandmarker();

    // Prediction loop using requestAnimationFrame
    let lastVideoTime = -1;
    const predictWebcam = () => {
      if (!runningRef.current || !handLandmarkerRef.current || !video || video.readyState < 2) {
        requestAnimationFrame(predictWebcam);
        return;
      }

      const nowInMs = performance.now();
      if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        handLandmarkerRef.current.detectForVideo(video, nowInMs, (results) => {
          processResults(results, ctx, canvas);
        });
      }

      requestAnimationFrame(predictWebcam);
    };

    // Cleanup
    return () => {
      window.removeEventListener('resize', setFullScreen);
      runningRef.current = false;
      if (handLandmarkerRef.current) {
        handLandmarkerRef.current.close();
      }
    };
  }, []);

  const processResults = (results, ctx, canvas) => {
    console.log('Processing results:', results); // Debug: Check if results are coming

    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Flip for mirror effect
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);

    // Draw video frame
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

    let detectedFingers = 0;
    let handHeight = 0;
    let newDistanceMessage = '';

    if (results.landmarks && results.landmarks.length > 0) {
      const landmarks = results.landmarks[0];

      // Draw landmarks
      drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 5 });
      drawLandmarks(ctx, landmarks, { color: '#FF0000', lineWidth: 2 });

      // Finger detection
      const fingerTips = [4, 8, 12, 16, 20];
      detectedFingers = fingerTips.filter(tip => landmarks[tip] && landmarks[tip].visibility > 0.8).length;

      // Distance estimation
      const wrist = landmarks[0];
      const middleTip = landmarks[12];
      if (wrist && middleTip) {
        const dx = (middleTip.x - wrist.x) * canvas.width;
        const dy = (middleTip.y - wrist.y) * canvas.height;
        handHeight = Math.sqrt(dx ** 2 + dy ** 2);
        const normalizedHeight = handHeight / canvas.height;
        if (normalizedHeight < 0.3) {
          newDistanceMessage = 'Hand too far - Move closer';
        } else if (normalizedHeight > 0.7) {
          newDistanceMessage = 'Hand too close - Move back';
        } else {
          newDistanceMessage = 'Good distance';
        }
      }
    }

    setFingerCount(detectedFingers);
    setDistanceMessage(newDistanceMessage);

    // Restore and draw text
    ctx.restore();
    ctx.font = '30px Arial';
    ctx.fillStyle = '#FFFFFF';
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 2;
    ctx.strokeText(`Detected Fingers: ${detectedFingers}/5`, 20, 50);
    ctx.fillText(`Detected Fingers: ${detectedFingers}/5`, 20, 50);
    ctx.strokeText(distanceMessage, 20, 100);
    ctx.fillText(distanceMessage, 20, 100);

    // Auto-capture
    if (detectedFingers === 5 && newDistanceMessage === 'Good distance') {
      const captureCanvas = document.createElement('canvas');
      captureCanvas.width = videoRef.current.videoWidth;
      captureCanvas.height = videoRef.current.videoHeight;
      captureCanvas.getContext('2d').drawImage(videoRef.current, 0, 0);
      const imageData = captureCanvas.toDataURL('image/jpeg');
      setCapturedImage(imageData);
      console.log('Captured image:', imageData);
      ctx.strokeText('Captured!', 20, 150);
      ctx.fillText('Captured!', 20, 150);
    }
  };

  return (
    <div style={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', overflow: 'hidden' }}>
      <video ref={videoRef} style={{ display: 'none' }} playsInline />
      <canvas ref={canvasRef} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }} />
    </div>
  );
}

export default CameraCapture;