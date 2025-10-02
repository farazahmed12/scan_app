import * as handpose from '@tensorflow-models/handpose';
import * as tf from '@tensorflow/tfjs';
import React, { useEffect, useRef, useState } from 'react';
import './NailKYCCamera.css';

export default function NailKYCCamera() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const capturedCanvasRef = useRef(null);
  
  const [model, setModel] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [fingersDetected, setFingersDetected] = useState(0);
  const [distanceStatus, setDistanceStatus] = useState('');
  const [capturedImage, setCapturedImage] = useState(null);
  const [countdown, setCountdown] = useState(null);

  useEffect(() => {
    let animationFrame;
    let countdownTimer;

    const setupCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { 
            facingMode: 'user',
            width: { ideal: 1920 },
            height: { ideal: 1080 }
          }
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current.play();
          };
        }
      } catch (err) {
        console.error('Camera error:', err);
      }
    };

    const loadModel = async () => {
      try {
        await tf.ready();
        const handposeModel = await handpose.load();
        setModel(handposeModel);
        setIsLoading(false);
      } catch (err) {
        console.error('Model error:', err);
      }
    };

    setupCamera();
    loadModel();

    return () => {
      if (animationFrame) cancelAnimationFrame(animationFrame);
      if (countdownTimer) clearTimeout(countdownTimer);
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    if (!model || !videoRef.current || isLoading) return;

    let animationFrame;
    let countdownTimer;
    let isCapturing = false;

    const detectHand = async () => {
      if (videoRef.current.readyState === 4) {
        const predictions = await model.estimateHands(videoRef.current);
        
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (predictions.length > 0) {
          const hand = predictions[0];
          const landmarks = hand.landmarks;
          
          // Calculate hand size to determine distance
          const handSize = calculateHandSize(landmarks);
          const distance = checkDistance(handSize, canvas.width);
          setDistanceStatus(distance);
          
          // Draw nails only
          drawNails(ctx, landmarks);
          
          const fingers = detectFingers(landmarks);
          setFingersDetected(fingers.length);
          
          // Auto-capture only if distance is perfect and all 5 fingers detected
          if (fingers.length === 5 && distance === 'PERFECT' && !isCapturing && !capturedImage) {
            isCapturing = true;
            
            let count = 3;
            setCountdown(count);
            
            const doCountdown = () => {
              if (count > 1) {
                count--;
                setCountdown(count);
                countdownTimer = setTimeout(doCountdown, 1000);
              } else {
                captureImage(landmarks);
                isCapturing = false;
                setCountdown(null);
              }
            };
            
            countdownTimer = setTimeout(doCountdown, 1000);
          } else if ((fingers.length < 5 || distance !== 'PERFECT') && countdownTimer) {
            clearTimeout(countdownTimer);
            setCountdown(null);
            isCapturing = false;
          }
        } else {
          setFingersDetected(0);
          setDistanceStatus('');
          if (countdownTimer) {
            clearTimeout(countdownTimer);
            setCountdown(null);
            isCapturing = false;
          }
        }
      }

      animationFrame = requestAnimationFrame(detectHand);
    };

    detectHand();

    return () => {
      if (animationFrame) cancelAnimationFrame(animationFrame);
      if (countdownTimer) clearTimeout(countdownTimer);
    };
  }, [model, isLoading, capturedImage]);

  const calculateHandSize = (landmarks) => {
    // Calculate distance between wrist (0) and middle finger tip (12)
    const [wristX, wristY] = landmarks[0];
    const [middleTipX, middleTipY] = landmarks[12];
    
    return Math.sqrt(
      Math.pow(middleTipX - wristX, 2) + Math.pow(middleTipY - wristY, 2)
    );
  };

  const checkDistance = (handSize, canvasWidth) => {
    // Calculate relative hand size compared to screen
    const relativeSize = handSize / canvasWidth;
    
    // Ideal nail capture range: hand should fill 30-50% of screen width
    if (relativeSize < 0.25) {
      return 'TOO FAR';
    } else if (relativeSize > 0.55) {
      return 'TOO CLOSE';
    } else {
      return 'PERFECT';
    }
  };

  const drawNails = (ctx, landmarks) => {
    // Draw only fingertips (nails) with enhanced visibility
    const fingertips = [4, 8, 12, 16, 20];
    const fingerNames = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'];
    
    fingertips.forEach((tip, index) => {
      const [x, y] = landmarks[tip];
      
      // Draw nail area
      ctx.strokeStyle = '#00ff00';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.arc(x, y, 20, 0, 2 * Math.PI);
      ctx.stroke();
      
      // Fill center
      ctx.fillStyle = 'rgba(0, 255, 0, 0.2)';
      ctx.fill();
      
      // Draw finger name
      ctx.fillStyle = '#00ff00';
      ctx.font = 'bold 18px Arial';
      ctx.fillText(fingerNames[index], x - 30, y - 30);
    });
  };

  const detectFingers = (landmarks) => {
    const fingertips = [4, 8, 12, 16, 20];
    const fingerBases = [2, 5, 9, 13, 17];
    const detectedFingers = [];
    
    fingertips.forEach((tip, index) => {
      const base = fingerBases[index];
      const [tipX, tipY] = landmarks[tip];
      const [baseX, baseY] = landmarks[base];
      
      const distance = Math.sqrt(
        Math.pow(tipX - baseX, 2) + Math.pow(tipY - baseY, 2)
      );
      
      if (distance > 50) {
        detectedFingers.push(index);
      }
    });
    
    return detectedFingers;
  };

  const captureImage = (landmarks) => {
    const video = videoRef.current;
    const capturedCanvas = capturedCanvasRef.current;
    const ctx = capturedCanvas.getContext('2d');
    
    capturedCanvas.width = video.videoWidth;
    capturedCanvas.height = video.videoHeight;
    
    ctx.drawImage(video, 0, 0);
    drawNails(ctx, landmarks);
    
    const imageData = capturedCanvas.toDataURL('image/png');
    setCapturedImage(imageData);
  };

  const resetCapture = () => {
    setCapturedImage(null);
    setDistanceStatus('');
  };

  if (capturedImage) {
    return (
      <div className="fullscreen-container">
        <img src={capturedImage} alt="Captured" className="captured-fullscreen" />
        <button onClick={resetCapture} className="recapture-btn">
          Capture Again
        </button>
      </div>
    );
  }

  return (
    <div className="fullscreen-container">
      <video
        ref={videoRef}
        className="fullscreen-video"
        autoPlay
        playsInline
        muted
      />
      <canvas
        ref={canvasRef}
        className="fullscreen-canvas"
      />
      <canvas ref={capturedCanvasRef} style={{ display: 'none' }} />
      
      {isLoading && (
        <div className="loading-fullscreen">
          <div className="loader"></div>
          <p>Loading AI Model...</p>
        </div>
      )}

      {!isLoading && (
        <>
          {/* Finger Count Display */}
          <div className="finger-counter">
            <div className="counter-circle">
              <span className="counter-number">{fingersDetected}</span>
              <span className="counter-label">/5</span>
            </div>
            <p className="counter-text">Fingers Detected</p>
          </div>

          {/* Distance Status */}
          {distanceStatus && (
            <div className={`distance-indicator ${distanceStatus.toLowerCase().replace(' ', '-')}`}>
              {distanceStatus === 'TOO FAR' && (
                <>
                  <span className="distance-icon">👋</span>
                  <span className="distance-text">Move Closer</span>
                </>
              )}
              {distanceStatus === 'TOO CLOSE' && (
                <>
                  <span className="distance-icon">✋</span>
                  <span className="distance-text">Move Back</span>
                </>
              )}
              {distanceStatus === 'PERFECT' && (
                <>
                  <span className="distance-icon">✓</span>
                  <span className="distance-text">Perfect Distance</span>
                </>
              )}
            </div>
          )}

          {/* Countdown */}
          {countdown && (
            <div className="countdown-fullscreen">
              {countdown}
            </div>
          )}

          {/* Instructions */}
          <div className="instructions-overlay">
            <p>Spread all 5 fingers clearly</p>
            <p>Focus on showing nails clearly</p>
          </div>
        </>
      )}
    </div>
  );
}