import * as handpose from '@tensorflow-models/handpose';
import * as tf from '@tensorflow/tfjs';
import { useEffect, useRef, useState } from 'react';
import './NailKYCCamera.css';

const calculateHandSize = (landmarks) => {
  const [wristX, wristY] = landmarks[0];
  const [middleTipX, middleTipY] = landmarks[12];
  
  return Math.sqrt(
    Math.pow(middleTipX - wristX, 2) + Math.pow(middleTipY - wristY, 2)
  );
};

const checkDistance = (handSize, canvasWidth) => {
  const relativeSize = handSize / canvasWidth;
  
  if (relativeSize < 0.35) {
    return 'TOO_FAR';
  } else if (relativeSize > 0.55) {
    return 'TOO_CLOSE';
  } else {
    return 'PERFECT';
  }
};

const calculateImageSharpness = (ctx, landmarks, canvasWidth, canvasHeight) => {
  const fingertips = [4, 8, 12, 16, 20];
  let totalVariance = 0;
  let sampleCount = 0;

  fingertips.forEach((tip) => {
    const [x, y] = landmarks[tip];
    const size = 30;
    const startX = Math.max(0, Math.floor(x - size / 2));
    const startY = Math.max(0, Math.floor(y - size / 2));
    const width = Math.min(size, canvasWidth - startX);
    const height = Math.min(size, canvasHeight - startY);

    if (width > 0 && height > 0) {
      const imageData = ctx.getImageData(startX, startY, width, height);
      const data = imageData.data;
      
      const grays = [];
      for (let i = 0; i < data.length; i += 4) {
        grays.push(data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114);
      }

      let sum = 0;
      for (let i = 1; i < grays.length; i++) {
        sum += Math.abs(grays[i] - grays[i - 1]);
      }
      totalVariance += sum / grays.length;
      sampleCount++;
    }
  });

  return sampleCount > 0 ? totalVariance / sampleCount : 0;
};

const checkSharpness = (sharpness) => {
  return sharpness > 8 ? 'SHARP' : 'BLURRY';
};

const drawNails = (ctx, landmarks) => {
  const fingertips = [4, 8, 12, 16, 20];
  const fingerNames = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'];
  
  fingertips.forEach((tip, index) => {
    const [x, y] = landmarks[tip];
    
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(x, y, 25, 0, 2 * Math.PI);
    ctx.stroke();
    
    ctx.fillStyle = 'rgba(0, 255, 0, 0.2)';
    ctx.fill();
    
    ctx.fillStyle = '#00ff00';
    ctx.font = 'bold 16px Inter';
    ctx.fillText(fingerNames[index], x - 25, y - 35);
  });
};

const checkNailSide = (landmarks) => {
  const palmPoints = [0, 5, 9, 13, 17];
  const tipPoints = [4, 8, 12, 16, 20];

  const avgPalmZ = palmPoints.reduce((sum, idx) => sum + landmarks[idx][2], 0) / palmPoints.length;
  const avgTipZ = tipPoints.reduce((sum, idx) => sum + landmarks[idx][2], 0) / tipPoints.length;

  const zDifference = avgPalmZ - avgTipZ;
  
  
  return zDifference > 0;
};

const detectFingers = (landmarks) => {
  const FINGER_LANDMARKS = [
    [4, 3, 2, 1],
    [8, 7, 6, 5],
    [12, 11, 10, 9],
    [16, 15, 14, 13],
    [20, 19, 18, 17]
  ];
  const detectedFingers = [];
  const wrist = landmarks[0];
  
  FINGER_LANDMARKS.forEach(([tipIndex, , pipIndex, mcpIndex], fingerId) => {
    const tip = landmarks[tipIndex];
    const pip = landmarks[pipIndex];
    const mcp = landmarks[mcpIndex];
    
    let isExtended = false;

    if (fingerId !== 0) {
      const distMCP_PIP = Math.sqrt(
        Math.pow(pip[0] - mcp[0], 2) + Math.pow(pip[1] - mcp[1], 2)
      );
      const distMCP_TIP = Math.sqrt(
        Math.pow(tip[0] - mcp[0], 2) + Math.pow(tip[1] - mcp[1], 2)
      );
      isExtended = distMCP_TIP > distMCP_PIP * 1.3;
    } else {
      const isProjectedOutwardZ = tip[2] < mcp[2];
      const distWrist_Tip = Math.sqrt(
        Math.pow(tip[0] - wrist[0], 2) + Math.pow(tip[1] - wrist[1], 2)
      );
      const distWrist_MCP = Math.sqrt(
        Math.pow(mcp[0] - wrist[0], 2) + Math.pow(mcp[1] - wrist[1], 2)
      );
      const isExtendedXY = distWrist_Tip > distWrist_MCP * 1.1;
      isExtended = isProjectedOutwardZ && isExtendedXY;
    }

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

export default function NailKYCCamera() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const capturedCanvasRef = useRef(null);
  
  const [model, setModel] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [fingersDetected, setFingersDetected] = useState(0);
  const [distanceStatus, setDistanceStatus] = useState('');
  const [sharpnessStatus, setSharpnessStatus] = useState('');
  const [capturedImage, setCapturedImage] = useState(null);
  const [countdown, setCountdown] = useState(null);
  const [showingNails, setShowingNails] = useState(true);

  useEffect(() => {
    let animationFrame;
    let countdownTimer;

    const setupCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { 
            // facingMode: 'user',
            facingMode: { exact: "environment" },
            width: { ideal: 1920 },
            height: { ideal: 1080 }
          }
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play();
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
    if (!model || !videoRef.current || isLoading || capturedImage) return;

    let animationFrame;
    let countdownTimer;
    let isCapturing = false;
    
    const videoElement = videoRef.current;
    const canvasElement = canvasRef.current;

    const detectHand = async () => {
      if (videoElement.readyState === 4 && canvasElement) {
        const predictions = await model.estimateHands(videoElement);
        
        const ctx = canvasElement.getContext('2d');
        if (!ctx) return;
        
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
        
        ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

        if (predictions.length > 0) {
          const landmarks = predictions[0].landmarks;
          
          const distance = checkDistance(calculateHandSize(landmarks), canvasElement.width);
          setDistanceStatus(distance);
          
          const sharpness = calculateImageSharpness(ctx, landmarks, canvasElement.width, canvasElement.height);
          const sharpnessResult = checkSharpness(sharpness);
          setSharpnessStatus(sharpnessResult);
          
          const fingers = detectFingers(landmarks);
          setFingersDetected(fingers.length);

          const isNailSide = checkNailSide(landmarks);
          setShowingNails(isNailSide);
          
          drawNails(ctx, landmarks);

          const isReady = fingers.length === 5 && distance === 'PERFECT' && isNailSide && sharpnessResult === 'SHARP';
          
          if (isReady && !isCapturing) {
            isCapturing = true;
            
            let count = 3;
            setCountdown(count);
            
            const doCountdown = () => {
              if (count > 1) {
                count--;
                setCountdown(count);
                countdownTimer = setTimeout(doCountdown, 1000);
              } else {
                captureImage();
                isCapturing = false;
                setCountdown(null);
              }
            };
            
            countdownTimer = setTimeout(doCountdown, 1000);
          } else if (!isReady && isCapturing) {
            clearTimeout(countdownTimer);
            setCountdown(null);
            isCapturing = false;
          }
        } else {
          setFingersDetected(0);
          setDistanceStatus('');
          setSharpnessStatus('');
          setShowingNails(true);
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
  }, [model, isLoading, capturedImage]);

  const captureImage = () => {
    const video = videoRef.current;
    const capturedCanvas = capturedCanvasRef.current;
    if (!video || !capturedCanvas) return;
    
    const ctx = capturedCanvas.getContext('2d');
    if (!ctx) return;
    
    capturedCanvas.width = video.videoWidth;
    capturedCanvas.height = video.videoHeight;
    
    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
    
    const imageData = capturedCanvas.toDataURL('image/png');
    setCapturedImage(imageData);
  };

  const resetCapture = () => {
    setCapturedImage(null);
    setDistanceStatus('');
    setSharpnessStatus('');
  };

  if (capturedImage) {
    return (
      <div className="nail-kyc-captured-container">
        <img src={capturedImage} alt="Captured" className="nail-kyc-captured-image" data-testid="img-captured" />
        <button
          onClick={resetCapture}
          data-testid="button-recapture"
          className="nail-kyc-recapture-button"
        >
          Capture Again
        </button>
      </div>
    );
  }

  return (
    <div className="nail-kyc-container">
      <video
        ref={videoRef}
        data-testid="video-camera"
        className="nail-kyc-video"
        autoPlay
        playsInline
        muted
      />
      <canvas
        ref={canvasRef}
        data-testid="canvas-overlay"
        className="nail-kyc-canvas"
      />
      <canvas ref={capturedCanvasRef} className="nail-kyc-canvas-hidden" />
      
      {isLoading && (
        <div className="nail-kyc-loading">
          <div className="nail-kyc-loader" data-testid="loader-ai" />
          <p className="nail-kyc-loading-text" data-testid="text-loading">Loading AI Model...</p>
        </div>
      )}

      {!isLoading && (
        <>
          <div className="nail-kyc-finger-counter" data-testid="counter-fingers">
            <div className={`nail-kyc-counter-circle ${fingersDetected === 5 ? 'complete' : ''}`}>
              <span className="nail-kyc-counter-number" data-testid="text-finger-count">{fingersDetected}</span>
              <span className="nail-kyc-counter-total" data-testid="text-finger-total">/5</span>
            </div>
            <p className="nail-kyc-counter-label" data-testid="label-fingers">Fingers Detected</p>
          </div>

          {distanceStatus && (
            <div 
              className={`nail-kyc-distance-indicator ${distanceStatus.toLowerCase().replace('_', '-')}`}
              data-testid={`indicator-distance-${distanceStatus.toLowerCase()}`}
            >
              <div className="nail-kyc-distance-content">
                {distanceStatus === 'TOO_FAR' && (
                  <>
                    <span className="nail-kyc-distance-icon" data-testid="icon-too-far">👋</span>
                    <span className="nail-kyc-distance-text" data-testid="text-distance-status">Move Closer</span>
                  </>
                )}
                {distanceStatus === 'TOO_CLOSE' && (
                  <>
                    <span className="nail-kyc-distance-icon" data-testid="icon-too-close">✋</span>
                    <span className="nail-kyc-distance-text" data-testid="text-distance-status">Move Back</span>
                  </>
                )}
                {distanceStatus === 'PERFECT' && (
                  <>
                    <span className="nail-kyc-distance-icon" data-testid="icon-perfect">✓</span>
                    <span className="nail-kyc-distance-text" data-testid="text-distance-status">Perfect Distance</span>
                  </>
                )}
              </div>
            </div>
          )}

          {!showingNails && distanceStatus !== 'TOO_FAR' && (
            <div className="nail-kyc-palm-warning" data-testid="warning-palm">
              <div className="nail-kyc-warning-content">
                <span className="nail-kyc-warning-icon" data-testid="icon-flip">🔄</span>
                <span className="nail-kyc-warning-text" data-testid="text-palm-warning">Flip Hand: Show Back of Hand (Nails Side)</span>
              </div>
            </div>
          )}

          {sharpnessStatus === 'BLURRY' && distanceStatus !== 'TOO_FAR' && (
            <div className="nail-kyc-blur-warning" data-testid="warning-blur">
              <div className="nail-kyc-blur-content">
                <span className="nail-kyc-blur-icon" data-testid="icon-blur">⚠️</span>
                <span className="nail-kyc-blur-text" data-testid="text-blur-warning">Image Blurry - Keep Hand Steady</span>
              </div>
            </div>
          )}

          {countdown && (
            <div className="nail-kyc-countdown-overlay" data-testid="overlay-countdown">
              <div className="nail-kyc-countdown-circle">
                <span className="nail-kyc-countdown-number" data-testid="text-countdown">{countdown}</span>
              </div>
            </div>
          )}

          <div className="nail-kyc-instructions" data-testid="instructions-overlay">
            <p className="nail-kyc-instruction-text" data-testid="text-instruction-1">• Spread all 5 fingers clearly</p>
            <p className="nail-kyc-instruction-text" data-testid="text-instruction-2">• Show back of hand (nails visible)</p>
            <p className="nail-kyc-instruction-text" data-testid="text-instruction-3">• Keep hand steady and in focus</p>
          </div>
        </>
      )}
    </div>
  );
}
