import * as handpose from '@tensorflow-models/handpose';
import * as tf from '@tensorflow/tfjs';
import React, { useEffect, useRef, useState } from 'react';
import './HandDetection.css';

const HandDetection = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [distanceStatus, setDistanceStatus] = useState('Click Start Camera to begin');
  const [handCount, setHandCount] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [autoCaptureTimeout, setAutoCaptureTimeout] = useState(null);

  useEffect(() => {
    initializeModel();
    return () => {
      // Cleanup
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
      if (autoCaptureTimeout) {
        clearTimeout(autoCaptureTimeout);
      }
    };
  }, []);

  const initializeModel = async () => {
    try {
      setIsLoading(true);
      console.log('Loading Handpose model...');
      
      // Set backend to WebGL for better performance
      await tf.setBackend('webgl');
      console.log('TensorFlow backend:', tf.getBackend());
      
      // Load handpose model
      const handposeModel = await handpose.load();
      setModel(handposeModel);
      console.log('Handpose model loaded successfully');
      
      setIsLoading(false);
    } catch (error) {
      console.error('Error loading handpose model:', error);
      setIsLoading(false);
    }
  };

  const startCamera = async () => {
    try {
      console.log('Starting camera...');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        },
        audio: false
      });
      
      videoRef.current.srcObject = stream;
      setIsCameraOn(true);
      setDistanceStatus('Bring hand into frame');
      
      videoRef.current.onloadeddata = () => {
        console.log('Video ready, starting detection');
        // Set canvas size to match video
        const video = videoRef.current;
        const canvas = canvasRef.current;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        console.log('Canvas size:', canvas.width, 'x', canvas.height);
        detectHands();
      };
      
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Cannot access camera. Please check permissions and make sure you are using HTTPS.');
    }
  };

  const detectHands = async () => {
    if (!model || !videoRef.current || !isCameraOn) {
      return;
    }

    try {
      // Check if video is ready
      if (videoRef.current.readyState !== 4) {
        requestAnimationFrame(detectHands);
        return;
      }

      const predictions = await model.estimateHands(videoRef.current);
      
      // Clear canvas
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      
      let fingerCount = 0;
      
      if (predictions.length > 0) {
        console.log('Hand detected!', predictions.length, 'hand(s)');
        // Draw hand landmarks
        drawHandLandmarks(predictions, ctx);
        
        // Count fingers for each hand
        predictions.forEach(prediction => {
          const count = countFingers(prediction.landmarks);
          fingerCount += count;
          console.log('Fingers counted:', count);
        });
        
        setHandCount(fingerCount);
        checkHandDistance(predictions, fingerCount);
      } else {
        setHandCount(0);
        setDistanceStatus('👋 Bring hand into frame');
        // Clear any existing timeout if no hand is detected
        if (autoCaptureTimeout) {
          clearTimeout(autoCaptureTimeout);
          setAutoCaptureTimeout(null);
        }
      }
    } catch (error) {
      console.error('Error detecting hands:', error);
    }

    // Continue detection
    requestAnimationFrame(detectHands);
  };

  const drawHandLandmarks = (predictions, ctx) => {
    ctx.fillStyle = '#00ff00';
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 3;

    predictions.forEach(prediction => {
      // Draw landmarks
      prediction.landmarks.forEach((landmark, index) => {
        ctx.beginPath();
        // Make fingertip landmarks larger
        const isFingertip = [4, 8, 12, 16, 20].includes(index);
        const radius = isFingertip ? 8 : 4;
        ctx.arc(landmark[0], landmark[1], radius, 0, 2 * Math.PI);
        ctx.fill();
        
        // Label fingertips with numbers
        if (isFingertip) {
          ctx.fillStyle = '#ff0000';
          ctx.font = '12px Arial';
          ctx.fillText(index === 4 ? 'T' : (index - 7).toString(), landmark[0] + 10, landmark[1] - 10);
          ctx.fillStyle = '#00ff00';
        }
      });

      // Draw connections
      drawHandConnections(prediction.landmarks, ctx);
    });
  };

  const drawHandConnections = (landmarks, ctx) => {
    // Hand connections
    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
      [0, 5], [5, 6], [6, 7], [7, 8], // Index finger
      [0, 9], [9, 10], [10, 11], [11, 12], // Middle finger
      [0, 13], [13, 14], [14, 15], [15, 16], // Ring finger
      [0, 17], [17, 18], [18, 19], [19, 20], // Pinky finger
      [5, 9], [9, 13], [13, 17] // Palm
    ];

    connections.forEach(connection => {
      const [start, end] = connection;
      ctx.beginPath();
      ctx.moveTo(landmarks[start][0], landmarks[start][1]);
      ctx.lineTo(landmarks[end][0], landmarks[end][1]);
      ctx.stroke();
    });
  };

  const countFingers = (landmarks) => {
    if (landmarks.length < 21) return 0;
    
    let count = 0;
    
    // Thumb - check if extended (simplified logic for mirrored video)
    if (landmarks[4][0] > landmarks[3][0]) count++;
    
    // Index finger - check if fingertip is above the middle joint
    if (landmarks[8][1] < landmarks[6][1]) count++;
    
    // Middle finger
    if (landmarks[12][1] < landmarks[10][1]) count++;
    
    // Ring finger
    if (landmarks[16][1] < landmarks[14][1]) count++;
    
    // Pinky finger
    if (landmarks[20][1] < landmarks[18][1]) count++;
    
    return count;
  };

  const checkHandDistance = (predictions, fingerCount) => {
    if (predictions.length === 0) return;

    const prediction = predictions[0];
    const landmarks = prediction.landmarks;
    
    if (landmarks.length < 21) {
      setDistanceStatus('Hand not fully visible');
      return;
    }

    // Calculate hand bounding box
    const xCoords = landmarks.map(lm => lm[0]);
    const yCoords = landmarks.map(lm => lm[1]);
    const minX = Math.min(...xCoords);
    const maxX = Math.max(...xCoords);
    const minY = Math.min(...yCoords);
    const maxY = Math.max(...yCoords);
    
    const width = maxX - minX;
    const height = maxY - minY;
    const handArea = width * height;

    const canvasArea = canvasRef.current.width * canvasRef.current.height;
    const coverage = handArea / canvasArea;

    console.log('Hand coverage:', coverage, 'Fingers:', fingerCount);

    // Draw bounding box for visual feedback
    const ctx = canvasRef.current.getContext('2d');
    ctx.strokeStyle = coverage >= 0.15 && coverage <= 0.3 && fingerCount === 5 ? '#00ff00' : '#ff0000';
    ctx.lineWidth = 3;
    ctx.strokeRect(minX, minY, width, height);

    // Visual feedback for distance
    if (coverage < 0.15) {
      setDistanceStatus('🔄 Move hand closer - Too far for nail measurement');
      if (autoCaptureTimeout) {
        clearTimeout(autoCaptureTimeout);
        setAutoCaptureTimeout(null);
      }
    } else if (coverage > 0.3) {
      setDistanceStatus('📏 Move hand farther - Too close for nail measurement');
      if (autoCaptureTimeout) {
        clearTimeout(autoCaptureTimeout);
        setAutoCaptureTimeout(null);
      }
    } else if (fingerCount === 5) {
      setDistanceStatus('✅ Perfect! All 5 nails visible. Keep steady...');
      
      // Auto-capture after 2 seconds of perfect position
      if (!autoCaptureTimeout) {
        const timeout = setTimeout(() => {
          if (fingerCount === 5 && coverage >= 0.15 && coverage <= 0.3) {
            captureImage();
          }
          setAutoCaptureTimeout(null);
        }, 2000);
        setAutoCaptureTimeout(timeout);
      }
    } else {
      setDistanceStatus(`👆 Show all 5 fingers - Currently ${fingerCount}/5 visible`);
      if (autoCaptureTimeout) {
        clearTimeout(autoCaptureTimeout);
        setAutoCaptureTimeout(null);
      }
    }
  };

  const captureImage = () => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    
    // Draw the current video frame
    ctx.drawImage(videoRef.current, 0, 0);
    
    const imageData = canvas.toDataURL('image/jpeg');
    console.log('Captured hand image for nail measurement');
    
    setDistanceStatus('✅ Nail image captured! Processing measurements...');
    
    // Show success message
    alert('🎉 Perfect! Hand image captured successfully!\n\nAll 5 nails are clearly visible and ready for measurement analysis.');
    
    // Here you would typically send imageData to your backend
    // For now, we'll just log it
    console.log('Image data ready for backend processing');
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      setIsCameraOn(false);
      setHandCount(0);
      setDistanceStatus('Camera stopped');
      if (autoCaptureTimeout) {
        clearTimeout(autoCaptureTimeout);
        setAutoCaptureTimeout(null);
      }
    }
  };

  if (isLoading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Loading Hand Detection Model...</p>
        <p className="loading-subtitle">This may take a few moments</p>
      </div>
    );
  }

  return (
    <div className="hand-detection-container">
      <div className="camera-container">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="camera-video"
        />
        <canvas
          ref={canvasRef}
          className="overlay-canvas"
        />
      </div>
      
      <div className="controls">
        {!isCameraOn ? (
          <button onClick={startCamera} className="start-button">
            Start Hand Detection
          </button>
        ) : (
          <div className="control-buttons">
            <button onClick={stopCamera} className="stop-button">
              Stop Camera
            </button>
            <button onClick={captureImage} className="capture-button">
              Capture Manually
            </button>
          </div>
        )}
      </div>
      
      <div className="status-panel">
        <div className="status-item">
          <strong>Fingers Visible:</strong> <span className="finger-count">{handCount}/5</span>
        </div>
        <div className="status-item">
          <strong>Guidance:</strong> <span className="distance-status">{distanceStatus}</span>
        </div>
        <div className="status-item">
          <strong>Goal:</strong> Capture clear images of all 5 nails
        </div>
      </div>

      <div className="instructions">
        <h3>📋 Instructions for Best Nail Capture:</h3>
        <ul>
          <li>✅ Keep hand steady and flat</li>
          <li>✅ Ensure all 5 fingers are visible</li>
          <li>✅ Good lighting on your hand</li>
          <li>✅ Position hand until guidance shows "Perfect!"</li>
          <li>✅ System will auto-capture when ready</li>
        </ul>
      </div>

      {/* Debug info */}
      <div className="debug-info">
        Camera: {isCameraOn ? 'On' : 'Off'} | Model: {model ? 'Loaded' : 'Loading'}
      </div>
    </div>
  );
};

export default HandDetection;