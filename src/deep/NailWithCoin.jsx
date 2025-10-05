// NailWithCoin.jsx
import * as handpose from "@tensorflow-models/handpose";
import * as tf from "@tensorflow/tfjs";
import React, { useCallback, useEffect, useRef, useState } from "react";

import coinFrameImage from "../coin.png";
import fingerFrameImage from "../fingerFrame.png";

const styles = `
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }

  .nail-coin-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-color: black;
    overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }

  .video {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  .canvas {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
  }

  .loading {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    text-align: center;
    color: white;
    z-index: 100;
    padding: 0 20px;
  }

  .loader-container {
    margin-bottom: 20px;
  }

  .loader {
    border: 6px solid rgba(255, 255, 255, 0.3);
    border-top: 6px solid white;
    border-radius: 50%;
    width: 60px;
    height: 60px;
    animation: spin 1s linear infinite;
    margin: 0 auto 15px;
  }

  .progress-bar {
    width: 200px;
    height: 8px;
    background-color: rgba(255, 255, 255, 0.3);
    border-radius: 4px;
    overflow: hidden;
    margin: 0 auto;
  }

  .progress-fill {
    height: 100%;
    background-color: #10b981;
    transition: width 0.3s ease;
  }

  .loading-text {
    font-size: 1.2rem;
    font-weight: bold;
    margin-top: 10px;
  }

  .status-bar {
    position: absolute;
    top: 20px;
    right: 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    z-index: 10;
    background: rgba(0, 0, 0, 0.8);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    min-width: 200px;
  }

  .status-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0;
    transition: all 0.3s ease;
  }

  .status-icon {
    font-size: 1.2rem;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.1);
    color: rgba(255, 255, 255, 0.5);
    font-weight: bold;
  }

  .status-active .status-icon {
    background: #10b981;
    color: white;
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
  }

  .status-text {
    color: rgba(255, 255, 255, 0.8);
    font-size: 0.9rem;
    font-weight: 500;
    flex: 1;
  }

  .status-active .status-text {
    color: white;
    font-weight: 600;
  }

  .distance-indicator {
    position: absolute;
    top: 45%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 10;
    padding: 15px 25px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.1rem;
    font-weight: bold;
    backdrop-filter: blur(10px);
  }

  .too-far {
    background: rgba(239, 68, 68, 0.9);
    color: white;
    box-shadow: 0 10px 30px rgba(239, 68, 68, 0.5);
  }

  .too-close {
    background: rgba(251, 191, 36, 0.9);
    color: white;
    box-shadow: 0 10px 30px rgba(251, 191, 36, 0.5);
  }

  .show-fingers {
    background: rgba(59, 130, 246, 0.9);
    color: white;
    box-shadow: 0 10px 30px rgba(59, 130, 246, 0.5);
  }

  .distance-text {
    font-size: 0.95rem;
  }

  .capture-btn {
    position: absolute;
    bottom: 30px;
    left: 50%;
    transform: translateX(-50%);
    color: white;
    font-size: 1.2rem;
    font-weight: bold;
    padding: 18px 50px;
    border: none;
    border-radius: 50px;
    cursor: pointer;
    z-index: 20;
    transition: all 0.3s ease;
    min-width: 250px;
    text-align: center;
  }

  .capture-btn-ready {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.5);
  }

  .capture-btn-disabled {
    background: rgba(107, 114, 128, 0.8);
    box-shadow: 0 10px 30px rgba(107, 114, 128, 0.3);
    cursor: not-allowed;
    opacity: 0.7;
  }

  .captured-image {
    width: 100%;
    height: 100%;
    object-fit: contain;
  }

  .capture-success {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(16, 185, 129, 0.95);
    color: white;
    padding: 20px 40px;
    border-radius: 15px;
    display: flex;
    align-items: center;
    gap: 15px;
    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.5);
  }

  .success-icon {
    font-size: 2.5rem;
  }

  .success-text {
    font-size: 1.3rem;
    font-weight: bold;
  }

  .recapture-btn {
    position: absolute;
    bottom: 30px;
    left: 50%;
    transform: translateX(-50%);
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-size: 1.2rem;
    font-weight: bold;
    padding: 18px 50px;
    border: none;
    border-radius: 50px;
    cursor: pointer;
    z-index: 20;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5);
    transition: all 0.3s ease;
    min-width: 250px;
    text-align: center;
  }

  .error-container {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    text-align: center;
    color: white;
    z-index: 100;
    padding: 0 30px;
  }

  .error-icon {
    font-size: 4rem;
    margin-bottom: 20px;
  }

  .error-text {
    font-size: 1.2rem;
    margin-bottom: 30px;
    line-height: 1.5;
  }

  .retry-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-size: 1.1rem;
    font-weight: bold;
    padding: 15px 40px;
    border: none;
    border-radius: 50px;
    cursor: pointer;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5);
  }
`;

// Finger detection logic from NailKYCCamera
const calculateFingerSize = (fingerTip, wrist) => {
  const distance = Math.sqrt(
    Math.pow(fingerTip[0] - wrist[0], 2) + Math.pow(fingerTip[1] - wrist[1], 2)
  );
  return distance;
};

const checkFingerDistance = (fingerSize, canvasHeight, history) => {
  const relativeSize = fingerSize / canvasHeight;
  const newHistory = [...history.slice(-4), relativeSize];
  const avgSize = newHistory.reduce((sum, val) => sum + val, 0) / newHistory.length;

  const TOO_FAR_THRESHOLD = 0.15;
  const TOO_CLOSE_THRESHOLD = 0.35;

  if (avgSize < TOO_FAR_THRESHOLD) return "TOO FAR";
  if (avgSize > TOO_CLOSE_THRESHOLD) return "TOO CLOSE";
  return "PERFECT";
};

const isPointInFrame = (point, frame) => {
  return (
    point[0] >= frame.x &&
    point[0] <= frame.x + frame.width &&
    point[1] >= frame.y &&
    point[1] <= frame.y + frame.height
  );
};

const detectIndexFinger = (landmarks) => {
  const indexTip = landmarks[8];
  const indexMid = landmarks[6];
  const indexBase = landmarks[5];

  const isTipAboveMiddle = indexTip[1] < indexMid[1];
  const isMiddleAboveBase = indexMid[1] < indexBase[1];

  return isTipAboveMiddle && isMiddleAboveBase;
};

// Circle detection validation from CircleDetect
const validateCircle = (grayImg, cx, cy, radius, cv) => {
  try {
    if (cx < radius || cy < radius || 
        cx + radius >= grayImg.cols || cy + radius >= grayImg.rows) {
      return false;
    }

    const numSamples = 16;
    let validPoints = 0;
    let edgeStrengthSum = 0;
    
    for (let i = 0; i < numSamples; i++) {
      const angle = (2 * Math.PI * i) / numSamples;
      const px = Math.round(cx + radius * Math.cos(angle));
      const py = Math.round(cy + radius * Math.sin(angle));
      
      const ix = Math.round(cx + (radius - 8) * Math.cos(angle));
      const iy = Math.round(cy + (radius - 8) * Math.sin(angle));
      
      const ox = Math.round(cx + (radius + 8) * Math.cos(angle));
      const oy = Math.round(cy + (radius + 8) * Math.sin(angle));
      
      if (px >= 0 && px < grayImg.cols && py >= 0 && py < grayImg.rows &&
          ix >= 0 && ix < grayImg.cols && iy >= 0 && iy < grayImg.rows &&
          ox >= 0 && ox < grayImg.cols && oy >= 0 && oy < grayImg.rows) {
        
        const perimeterPixel = grayImg.ucharPtr(py, px)[0];
        const innerPixel = grayImg.ucharPtr(iy, ix)[0];
        const outerPixel = grayImg.ucharPtr(oy, ox)[0];
        
        const innerGradient = Math.abs(perimeterPixel - innerPixel);
        const outerGradient = Math.abs(perimeterPixel - outerPixel);
        const totalGradient = innerGradient + outerGradient;
        
        edgeStrengthSum += totalGradient;
        
        if (totalGradient > 40) {
          validPoints++;
        }
      }
    }
    
    const circleScore = validPoints / numSamples;
    const avgEdgeStrength = edgeStrengthSum / numSamples;
    
    return (circleScore >= 0.55 && avgEdgeStrength >= 20) || avgEdgeStrength >= 35;
  } catch (err) {
    return false;
  }
};

export default function NailWithCoin() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const capturedCanvasRef = useRef(null);
  const detectionLoopRef = useRef(null);
  const streamRef = useRef(null);
  const isDetectionActiveRef = useRef(false);
  const fingerFrameImageRef = useRef(null);
  const coinFrameImageRef = useRef(null);

  const [model, setModel] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [distanceStatus, setDistanceStatus] = useState("");
  const [capturedImage, setCapturedImage] = useState(null);
  const [fingerInFrame, setFingerInFrame] = useState(false);
  const [circleDetected, setCircleDetected] = useState(false);
  const [sizeHistory, setSizeHistory] = useState([]);
  const [error, setError] = useState(null);
  const [isOpenCVLoaded, setIsOpenCVLoaded] = useState(false);

  const [fingerFrame, setFingerFrame] = useState({
    x: 0, y: 0, width: 352, height: 222
  });

  const [coinFrame, setCoinFrame] = useState({
    x: 0, y: 0, width: 200, height: 200
  });

  const [isFingerFrameLoaded, setIsFingerFrameLoaded] = useState(false);
  const [isCoinFrameLoaded, setIsCoinFrameLoaded] = useState(false);

  // Load OpenCV
  useEffect(() => {
    if (typeof window.cv !== 'undefined' && window.cv.Mat) {
      setIsOpenCVLoaded(true);
      return;
    }

    const existingScript = document.querySelector('script[src*="opencv"]');
    if (existingScript) {
      const checkCV = setInterval(() => {
        if (typeof window.cv !== 'undefined' && window.cv.Mat) {
          setIsOpenCVLoaded(true);
          clearInterval(checkCV);
        }
      }, 100);
      setTimeout(() => clearInterval(checkCV), 10000);
      return;
    }

    const script = document.createElement('script');
    script.src = 'https://docs.opencv.org/4.5.2/opencv.js';
    script.async = true;
    
    script.onload = () => {
      const checkCV = setInterval(() => {
        if (typeof window.cv !== 'undefined' && window.cv.Mat) {
          setIsOpenCVLoaded(true);
          clearInterval(checkCV);
        }
      }, 100);
      setTimeout(() => clearInterval(checkCV), 10000);
    };
    
    script.onerror = () => {
      const altScript = document.createElement('script');
      altScript.src = 'https://cdn.jsdelivr.net/npm/opencv.js@1.2.1/opencv.js';
      altScript.async = true;
      
      altScript.onload = () => {
        const checkCV = setInterval(() => {
          if (typeof window.cv !== 'undefined' && window.cv.Mat) {
            setIsOpenCVLoaded(true);
            clearInterval(checkCV);
          }
        }, 100);
        setTimeout(() => clearInterval(checkCV), 10000);
      };
      
      document.body.appendChild(altScript);
    };
    
    document.body.appendChild(script);
  }, []);

  // Preload images
  useEffect(() => {
    const fingerImg = new Image();
    fingerImg.src = fingerFrameImage;
    fingerImg.onload = () => {
      fingerFrameImageRef.current = fingerImg;
      setIsFingerFrameLoaded(true);
    };

    const coinImg = new Image();
    coinImg.src = coinFrameImage;
    coinImg.onload = () => {
      coinFrameImageRef.current = coinImg;
      setIsCoinFrameLoaded(true);
    };
  }, []);

  // Update frame positions
  useEffect(() => {
    const updateFrames = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const fingerFrameWidth = 352;
      const fingerFrameHeight = 222;
      const coinFrameSize = 200;
      const gap = 40;

      const totalWidth = fingerFrameWidth + gap + coinFrameSize;
      const startX = (canvas.width - totalWidth) / 2;
      const centerY = canvas.height / 2;

      setFingerFrame({
        x: startX,
        y: centerY - fingerFrameHeight / 2,
        width: fingerFrameWidth,
        height: fingerFrameHeight,
      });

      setCoinFrame({
        x: startX + fingerFrameWidth + gap,
        y: centerY - coinFrameSize / 2,
        width: coinFrameSize,
        height: coinFrameSize,
      });
    };

    updateFrames();
  }, []);

  // Detect coin/circle in frame using OpenCV
  const detectCircleInFrame = (ctx, frame) => {
    if (!isOpenCVLoaded || typeof window.cv === 'undefined') return false;

    try {
      const cv = window.cv;
      const imageData = ctx.getImageData(frame.x, frame.y, frame.width, frame.height);
      
      const src = cv.matFromImageData(imageData);
      const gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
      
      cv.medianBlur(gray, gray, 5);
      
      const circles = new cv.Mat();
      cv.HoughCircles(gray, circles, cv.HOUGH_GRADIENT, 1.2, 45, 90, 40, 18, 80);

      let circleFound = false;
      if (circles.cols > 0) {
        for (let i = 0; i < circles.cols; ++i) {
          const cx = circles.data32F[i * 3];
          const cy = circles.data32F[i * 3 + 1];
          const radius = circles.data32F[i * 3 + 2];
          
          if (validateCircle(gray, cx, cy, radius, cv)) {
            circleFound = true;
            break;
          }
        }
      }
      
      src.delete();
      gray.delete();
      circles.delete();
      
      return circleFound;
    } catch (err) {
      console.error("Circle detection error:", err);
      return false;
    }
  };

  const setupCamera = useCallback(async (retryCount = 0) => {
    try {
      setLoadingProgress(20);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280, max: 1920 },
          height: { ideal: 720, max: 1080 },
          frameRate: { ideal: 30, max: 30 },
        },
      });

      streamRef.current = stream;
      setLoadingProgress(40);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await new Promise((resolve, reject) => {
          videoRef.current.onloadedmetadata = () => {
            videoRef.current.play().then(resolve).catch(reject);
          };
          setTimeout(() => reject(new Error("Video load timeout")), 10000);
        });
        setLoadingProgress(60);
      }
    } catch (err) {
      console.error("Camera setup error:", err);
      if (retryCount < 3) {
        setTimeout(() => setupCamera(retryCount + 1), 2000);
      } else {
        setError("Camera access failed. Please check permissions.");
      }
    }
  }, []);

  const loadModel = useCallback(async () => {
    try {
      setLoadingProgress(70);
      await tf.ready();
      await tf.setBackend("webgl");

      setLoadingProgress(85);
      const handposeModel = await handpose.load();
      setModel(handposeModel);
      setLoadingProgress(100);
      setIsLoading(false);
    } catch (err) {
      console.error("Model loading error:", err);
      setError("AI model failed to load. Please restart.");
    }
  }, []);

  const detectHand = useCallback(async () => {
    if (!model || !videoRef.current || !isDetectionActiveRef.current || capturedImage) {
      return;
    }

    try {
      if (videoRef.current.readyState !== 4) {
        detectionLoopRef.current = setTimeout(detectHand, 100);
        return;
      }

      const predictions = await model.estimateHands(videoRef.current);
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");

      if (canvas.width !== videoRef.current.videoWidth) {
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        
        // Update frames with new canvas size
        const fingerFrameWidth = 352;
        const fingerFrameHeight = 222;
        const coinFrameSize = 200;
        const gap = 40;

        const totalWidth = fingerFrameWidth + gap + coinFrameSize;
        const startX = (canvas.width - totalWidth) / 2;
        const centerY = canvas.height / 2;

        setFingerFrame({
          x: startX,
          y: centerY - fingerFrameHeight / 2,
          width: fingerFrameWidth,
          height: fingerFrameHeight,
        });

        setCoinFrame({
          x: startX + fingerFrameWidth + gap,
          y: centerY - coinFrameSize / 2,
          width: coinFrameSize,
          height: coinFrameSize,
        });
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw frames
      if (fingerFrameImageRef.current) {
        ctx.drawImage(
          fingerFrameImageRef.current,
          fingerFrame.x,
          fingerFrame.y,
          fingerFrame.width,
          fingerFrame.height
        );
      }

      if (coinFrameImageRef.current) {
        ctx.drawImage(
          coinFrameImageRef.current,
          coinFrame.x,
          coinFrame.y,
          coinFrame.width,
          coinFrame.height
        );
      }

      // Detect circle in coin frame
      const circlePresent = detectCircleInFrame(ctx, coinFrame);
      setCircleDetected(circlePresent);

      let fingerValid = false;
      let distanceOk = false;

      if (predictions.length > 0) {
        const hand = predictions[0];
        if (hand.handInViewConfidence > 0.75) {
          const landmarks = hand.landmarks;
          const isIndexExtended = detectIndexFinger(landmarks);

          if (isIndexExtended) {
            const indexTip = landmarks[8];
            const wrist = landmarks[0];

            const inFrame = isPointInFrame(indexTip, fingerFrame);
            setFingerInFrame(inFrame);
            fingerValid = inFrame;

            if (inFrame) {
              const fingerSize = calculateFingerSize(indexTip, wrist);
              const distance = checkFingerDistance(fingerSize, canvas.height, sizeHistory);
              setDistanceStatus(distance);
              setSizeHistory((prev) => [...prev.slice(-4), fingerSize / canvas.height]);
              distanceOk = distance === "PERFECT";

              // Draw finger tip
              ctx.beginPath();
              ctx.arc(indexTip[0], indexTip[1], 15, 0, 2 * Math.PI);
              ctx.fillStyle = distance === "PERFECT" ? "#10b981" : "#f59e0b";
              ctx.fill();
              ctx.strokeStyle = "white";
              ctx.lineWidth = 3;
              ctx.stroke();
            } else {
              setDistanceStatus("FINGER NOT IN FRAME");
            }
          } else {
            setFingerInFrame(false);
            setDistanceStatus("SHOW INDEX FINGER");
          }
        } else {
          setFingerInFrame(false);
          setDistanceStatus("");
        }
      } else {
        setFingerInFrame(false);
        setDistanceStatus("");
        setSizeHistory([]);
      }
    } catch (err) {
      console.error("Detection error:", err);
    }

    detectionLoopRef.current = setTimeout(detectHand, 150);
  }, [model, capturedImage, sizeHistory, fingerFrame, coinFrame, isOpenCVLoaded]);

  const handleCapture = useCallback(() => {
    if (!fingerInFrame || !circleDetected || distanceStatus !== "PERFECT") {
      return;
    }

    try {
      const canvas = canvasRef.current;
      const capturedCanvas = capturedCanvasRef.current;
      const ctx = capturedCanvas.getContext("2d");
      const video = videoRef.current;

      capturedCanvas.width = canvas.width;
      capturedCanvas.height = canvas.height;

      ctx.drawImage(video, 0, 0, capturedCanvas.width, capturedCanvas.height);

      const imageData = capturedCanvas.toDataURL("image/jpeg", 0.95);
      setCapturedImage(imageData);
      isDetectionActiveRef.current = false;

      // Log the captured image data (for backend integration)
      console.log("Captured Image Data:", {
        imageData: imageData.substring(0, 100) + "...", // Log first 100 chars
        dimensions: {
          width: capturedCanvas.width,
          height: capturedCanvas.height,
        },
        timestamp: new Date().toISOString(),
      });

    } catch (err) {
      console.error("Capture error:", err);
    }
  }, [fingerInFrame, circleDetected, distanceStatus]);

  const resetCapture = useCallback(async () => {
    setCapturedImage(null);
    setDistanceStatus("");
    setFingerInFrame(false);
    setCircleDetected(false);
    setSizeHistory([]);
    setError(null);

    await setupCamera();
    isDetectionActiveRef.current = true;
  }, [setupCamera]);

  useEffect(() => {
    setupCamera();
    loadModel();

    return () => {
      if (detectionLoopRef.current) {
        clearTimeout(detectionLoopRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    if (model && !isLoading && !capturedImage && isFingerFrameLoaded && isCoinFrameLoaded) {
      isDetectionActiveRef.current = true;
      detectHand();
    }

    return () => {
      isDetectionActiveRef.current = false;
      if (detectionLoopRef.current) {
        clearTimeout(detectionLoopRef.current);
      }
    };
  }, [model, isLoading, capturedImage, detectHand, isFingerFrameLoaded, isCoinFrameLoaded]);

  if (error) {
    return (
      <div className="nail-coin-container">
        <style>{styles}</style>
        <div className="error-container">
          <div className="error-icon">⚠️</div>
          <p className="error-text">{error}</p>
          <button onClick={resetCapture} className="retry-btn">
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (capturedImage) {
    return (
      <div className="nail-coin-container">
        <style>{styles}</style>
        <img src={capturedImage} alt="Captured" className="captured-image" />
        <div className="capture-success">
          <span className="success-icon">✓</span>
          <span className="success-text">Image Captured Successfully!</span>
        </div>
        <button onClick={resetCapture} className="recapture-btn">
          Capture Again
        </button>
      </div>
    );
  }

  const isReadyToCapture = fingerInFrame && circleDetected && distanceStatus === "PERFECT";

  return (
    <div className="nail-coin-container">
      <style>{styles}</style>
      <video ref={videoRef} className="video" autoPlay playsInline muted />
      <canvas ref={canvasRef} className="canvas" />
      <canvas ref={capturedCanvasRef} style={{ display: "none" }} />

      {isLoading && (
        <div className="loading">
          <div className="loader-container">
            <div className="loader"></div>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${loadingProgress}%` }} />
            </div>
          </div>
          <p className="loading-text">
            {loadingProgress < 60
              ? "Initializing Camera..."
              : loadingProgress < 100
              ? "Loading AI Model..."
              : "Ready!"}
          </p>
        </div>
      )}

      {!isLoading && (
        <>
          <div className="status-bar">
            <div className={`status-item ${fingerInFrame ? 'status-active' : ''}`}>
              <span className="status-icon">{fingerInFrame ? "✓" : "○"}</span>
              <span className="status-text">Finger Detected</span>
            </div>
            <div className={`status-item ${circleDetected ? 'status-active' : ''}`}>
              <span className="status-icon">{circleDetected ? "✓" : "○"}</span>
              <span className="status-text">Circle Detected</span>
            </div>
            <div className={`status-item ${distanceStatus === 'PERFECT' ? 'status-active' : ''}`}>
              <span className="status-icon">{distanceStatus === 'PERFECT' ? "✓" : "○"}</span>
              <span className="status-text">Distance OK</span>
            </div>
          </div>

          {distanceStatus && distanceStatus !== "PERFECT" && (
            <div className={`distance-indicator ${
              distanceStatus === "TOO FAR" ? "too-far" :
              distanceStatus === "TOO CLOSE" ? "too-close" :
              "show-fingers"
            }`}>
              <span className="distance-text">{distanceStatus}</span>
            </div>
          )}

          <button
            onClick={handleCapture}
            disabled={!isReadyToCapture}
            className={`capture-btn ${isReadyToCapture ? 'capture-btn-ready' : 'capture-btn-disabled'}`}
          >
            {isReadyToCapture ? "📸 Capture Image" : "Align Finger & Coin"}
          </button>
        </>
      )}
    </div>
  );
}