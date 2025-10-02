import * as handpose from '@tensorflow-models/handpose';
import * as tf from '@tensorflow/tfjs';
import React, { useEffect, useRef, useState } from 'react';
import './NailKYCCamera.css';

// --- Utility Functions (moved outside for clarity) ---

const calculateHandSize = (landmarks) => {
  // Calculate distance between wrist (0) and middle finger tip (12)
  const [wristX, wristY] = landmarks[0];
  const [middleTipX, middleTipY] = landmarks[12];
  
  return Math.sqrt(
    Math.pow(middleTipX - wristX, 2) + Math.pow(middleTipY - wristY, 2)
  );
};

const checkDistance = (handSize, canvasWidth) => {
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
  // Indices: [Tip, DIP, PIP, MCP]
  const FINGER_LANDMARKS = [
      [4, 3, 2, 1],   // Thumb
      [8, 7, 6, 5],   // Index
      [12, 11, 10, 9],  // Middle
      [16, 15, 14, 13], // Ring
      [20, 19, 18, 17]  // Pinky
  ];
  const detectedFingers = [];
  const wrist = landmarks[0];
  
  FINGER_LANDMARKS.forEach(([tipIndex, , pipIndex, mcpIndex], fingerId) => {
      const tip = landmarks[tipIndex];
      const pip = landmarks[pipIndex];
      const mcp = landmarks[mcpIndex]; // Base Joint

      let isExtended = false;

      if (fingerId !== 0) { // Index, Middle, Ring, Pinky (Fingers 1-4)
          // Distance from MCP (Base) to PIP (Middle Joint)
          const distMCP_PIP = Math.sqrt(
              Math.pow(pip[0] - mcp[0], 2) + Math.pow(pip[1] - mcp[1], 2)
          );

          // Distance from MCP (Base) to Tip
          const distMCP_TIP = Math.sqrt(
              Math.pow(tip[0] - mcp[0], 2) + Math.pow(tip[1] - mcp[1], 2)
          );

          // Tip distance must be at least 1.3 times the middle joint distance to be considered extended
          isExtended = distMCP_TIP > distMCP_PIP * 1.3; 
      } else { // Thumb (Finger 0)
          
          // A) Z-Depth Check: Tip (tip[2]) must be closer to the camera (lower Z) than the base (mcp[2]).
          // This prevents detecting a tucked-back thumb.
          const isProjectedOutwardZ = tip[2] < mcp[2]; 

          // B) X-Y Extension Check: Tip must be significantly further from the wrist than the base joint.
          const distWrist_Tip = Math.sqrt(
              Math.pow(tip[0] - wrist[0], 2) + Math.pow(tip[1] - wrist[1], 2)
          );
          const distWrist_MCP = Math.sqrt(
              Math.pow(mcp[0] - wrist[0], 2) + Math.pow(mcp[1] - wrist[1], 2)
          );
          const isExtendedXY = distWrist_Tip > distWrist_MCP * 1.1; // 10% tolerance

          isExtended = isProjectedOutwardZ && isExtendedXY;
      }

      // 2. Visibility Check (Tip must be a minimum distance from the wrist to be visible)
      const wristToTip = Math.sqrt(
          Math.pow(tip[0] - wrist[0], 2) + Math.pow(tip[1] - wrist[1], 2)
      );
      const isVisible = wristToTip > 100; 

      if (isExtended && isVisible) {
          detectedFingers.push(fingerId);
      }
  });
  
  return detectedFingers;
};


// --- React Component ---

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
        // Request higher resolution for better nail detail capture
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
        // 1. Detect Hand
        const predictions = await model.estimateHands(videoRef.current);
        
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        
        // Match canvas to video size
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (predictions.length > 0) {
          const hand = predictions[0];
          const landmarks = hand.landmarks;
          
          // 2. Distance Check
          const handSize = calculateHandSize(landmarks);
          const distance = checkDistance(handSize, canvas.width);
          setDistanceStatus(distance);
          
          // 3. Finger Extension Check (Using the enhanced logic)
          const fingers = detectFingers(landmarks);
          setFingersDetected(fingers.length);
          
          // 4. Drawing Overlays
          drawNails(ctx, landmarks);
          
          // 5. Auto-capture logic
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
          } else if ((fingers.length < 5 || distance !== 'PERFECT') && isCapturing) {
            // Reset countdown if conditions are lost
            clearTimeout(countdownTimer);
            setCountdown(null);
            isCapturing = false;
          }
        } else {
          setFingersDetected(0);
          setDistanceStatus('');
          if (isCapturing) {
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
  }, [model, isLoading, capturedImage]); // Dependency on capturedImage prevents re-detection

  const captureImage = (landmarks) => {
    const video = videoRef.current;
    const capturedCanvas = capturedCanvasRef.current;
    const ctx = capturedCanvas.getContext('2d');
    
    capturedCanvas.width = video.videoWidth;
    capturedCanvas.height = video.videoHeight;
    
    // Draw the image without flipping for proper capture
    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight); 
    
    // Draw the highlights/markers on the captured image
    // You might want to skip drawing overlays on the final image depending on your requirement
    // drawNails(ctx, landmarks); 
    
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