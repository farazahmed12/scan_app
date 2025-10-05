import * as handpose from "@tensorflow-models/handpose";
import * as tf from "@tensorflow/tfjs";
import React, { useCallback, useEffect, useRef, useState } from "react";

import coinFrameImage from "../coin.png";
import fingerFrameImage from "../fingerFrame.png";

const calculateFingerSize = (fingerTip, wrist) => {
  const distance = Math.sqrt(
    Math.pow(fingerTip[0] - wrist[0], 2) + Math.pow(fingerTip[1] - wrist[1], 2)
  );
  return distance;
};

const checkFingerDistance = (fingerSize, canvasHeight, history) => {
  const relativeSize = fingerSize / canvasHeight;
  const newHistory = [...history.slice(-4), relativeSize];
  const avgSize =
    newHistory.reduce((sum, val) => sum + val, 0) / newHistory.length;


    const TOO_FAR_THRESHOLD = 0.15;
    const TOO_CLOSE_THRESHOLD = 0.4;
  
    if (avgSize < TOO_FAR_THRESHOLD) return "TOO FAR";
    if (avgSize > TOO_CLOSE_THRESHOLD) return "TOO CLOSE";
    return "PERFECT";

};


const checkDistance = (fingerSize, canvasHeight, history) => {
    const relativeSize = fingerSize / canvasHeight;
    const newHistory = [...history.slice(-4), relativeSize];
    const avgSize =
      newHistory.reduce((sum, val) => sum + val, 0) / newHistory.length;
  
      return avgSize
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
  const detectionHistoryRef = useRef([]);

  const [handposeModel, setHandposeModel] = useState(null);
  const [isOpenCVLoaded, setIsOpenCVLoaded] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [distanceStatus, setDistanceStatus] = useState("");
  const [capturedImage, setCapturedImage] = useState(null);
  const [fingerInFrame, setFingerInFrame] = useState(false);
  const [coinDetected, setCoinDetected] = useState(false);
  const [sizeHistory, setSizeHistory] = useState([]);
  const [error, setError] = useState(null);
  const [dValue, setDValue] = useState(0);

  const FINGER_FRAME_WIDTH  = 400
  const FINGER_FRAME_HEIGHT = 500
  const COIN_FRAME_HEIGHT = 150
  const COIN_FRAME_WIDTH = 250

  
  // Larger finger frame, smaller coin frame
  const [fingerFrame, setFingerFrame] = useState({
    x: 0,
    y: 0,
    width: FINGER_FRAME_WIDTH,  
    height: FINGER_FRAME_HEIGHT, 
  });
  const [coinFrame, setCoinFrame] = useState({
    x: 0,
    y: 0,
    width: COIN_FRAME_WIDTH,
    height: COIN_FRAME_HEIGHT
  });
  
  const [isImagesLoaded, setIsImagesLoaded] = useState(false);
  const [isFlashOn, setIsFlashOn] = useState(false);

  const HISTORY_LENGTH = 8;
  const STABILITY_THRESHOLD = 6;

  const turnOnFlash = async () => {
    if (isFlashOn) return;

    if (window.ReactNativeWebView && window.toggleNativeTorch) {
      window.toggleNativeTorch();
      setIsFlashOn(true);
    } else {
      try {
        const stream = videoRef.current?.srcObject;
        if (!stream) return;

        const track = stream.getVideoTracks()[0];
        const capabilities = track.getCapabilities();

        if (capabilities.torch) {
          await track.applyConstraints({
            advanced: [{ torch: true }],
          });
          setIsFlashOn(true);
        }
      } catch (err) {
        console.error("Flash on error:", err);
      }
    }
  };

  const setupCamera = useCallback(async (retryCount = 0) => {
    try {
      setLoadingProgress(20);

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
        // facingMode: { exact: "environment" },
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

  const loadModels = useCallback(async () => {
    try {
      setLoadingProgress(70);
      
      await tf.ready();
      await tf.setBackend("webgl");

      setLoadingProgress(80);
      const handModel = await handpose.load();
      setHandposeModel(handModel);

      setLoadingProgress(90);

      if (typeof window.cv === 'undefined' || !window.cv.Mat) {
        await new Promise((resolve, reject) => {
          const checkCV = setInterval(() => {
            if (typeof window.cv !== 'undefined' && window.cv.Mat) {
              setIsOpenCVLoaded(true);
              clearInterval(checkCV);
              resolve();
            }
          }, 100);
          
          setTimeout(() => {
            clearInterval(checkCV);
            reject(new Error("OpenCV timeout"));
          }, 10000);
        });
      } else {
        setIsOpenCVLoaded(true);
      }

      setLoadingProgress(100);
      setIsLoading(false);
      
      // Turn on flash after models are loaded
      setTimeout(() => {
        turnOnFlash();
      }, 500);
    } catch (err) {
      console.error("Model loading error:", err);
      setError("AI models failed to load. Please restart.");
    }
  }, []);

  useEffect(() => {
    if (typeof window.cv === 'undefined') {
      const existingScript = document.querySelector('script[src*="opencv"]');
      if (!existingScript) {
        const script = document.createElement('script');
        script.src = 'https://docs.opencv.org/4.5.2/opencv.js';
        script.async = true;
        script.id = 'opencv-script';
        
        script.onload = () => {
          console.log("OpenCV script loaded");
        };
        
        script.onerror = () => {
          console.error("Failed to load OpenCV.js");
        };
        
        document.body.appendChild(script);
      }
    }
  }, []);

  useEffect(() => {
    let loaded = 0;
    const checkLoaded = () => {
      loaded++;
      if (loaded === 2) setIsImagesLoaded(true);
    };

    const fingerImg = new Image();
    fingerImg.src = fingerFrameImage;
    fingerImg.onload = () => {
      fingerFrameImageRef.current = fingerImg;
      checkLoaded();
    };
    fingerImg.onerror = () => console.error("Failed to load finger frame");

    const coinImg = new Image();
    coinImg.src = coinFrameImage;
    coinImg.onload = () => {
      coinFrameImageRef.current = coinImg;
      checkLoaded();
    };
    coinImg.onerror = () => console.error("Failed to load coin frame");
  }, []);

  useEffect(() => {
    const updateFrames = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      // Updated sizes: larger finger frame, smaller coin frame
      const fingerFrameWidth = FINGER_FRAME_WIDTH;
      const fingerFrameHeight = FINGER_FRAME_HEIGHT;
      const coinFrameSize = COIN_FRAME_WIDTH;
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

  const detectCoin = useCallback((video, frame) => {
    if (!isOpenCVLoaded || typeof window.cv === 'undefined' || !video) return false;

    try {
      const cv = window.cv;
      
      // Create a temporary canvas to get clean video frame data
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = video.videoWidth;
      tempCanvas.height = video.videoHeight;
      const tempCtx = tempCanvas.getContext('2d');
      tempCtx.drawImage(video, 0, 0);
      
      // Get image data from the coin frame region
      const imageData = tempCtx.getImageData(frame.x, frame.y, frame.width, frame.height);
      const src = cv.matFromImageData(imageData);
      
      const gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
      cv.medianBlur(gray, gray, 5);
      
      const circles = new cv.Mat();
      cv.HoughCircles(gray, circles, cv.HOUGH_GRADIENT, 1.2, 45, 90, 40, 18, 80);

      let detected = false;
      if (circles.cols > 0) {
        for (let i = 0; i < circles.cols; i++) {
          const cx = circles.data32F[i * 3];
          const cy = circles.data32F[i * 3 + 1];
          const radius = circles.data32F[i * 3 + 2];
          
          if (validateCircle(gray, cx, cy, radius, cv)) {
            detected = true;
            break;
          }
        }
      }

      detectionHistoryRef.current.push(detected);
      if (detectionHistoryRef.current.length > HISTORY_LENGTH) {
        detectionHistoryRef.current.shift();
      }

      const positiveDetections = detectionHistoryRef.current.filter(d => d).length;
      const stable = positiveDetections >= STABILITY_THRESHOLD;

      src.delete();
      gray.delete();
      circles.delete();

      return stable;
    } catch (err) {
      console.error("Coin detection error:", err);
      return false;
    }
  }, [isOpenCVLoaded]);

  const detectHand = useCallback(async () => {
    if (
      !handposeModel ||
      !videoRef.current ||
      !isDetectionActiveRef.current ||
      capturedImage ||
      !isImagesLoaded
    ) {
      return;
    }

    try {
      if (videoRef.current.readyState !== 4) {
        detectionLoopRef.current = setTimeout(detectHand, 100);
        return;
      }

      const predictions = await handposeModel.estimateHands(videoRef.current);
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");

      if (canvas.width !== videoRef.current.videoWidth) {
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;

        // Updated sizes for responsive layout
        const fingerFrameWidth = FINGER_FRAME_WIDTH;
        const fingerFrameHeight = FINGER_FRAME_HEIGHT;
        const coinFrameSize = COIN_FRAME_WIDTH
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

      // Draw dark overlay over entire canvas
      ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Clear the finger frame area (make it transparent)
      ctx.clearRect(fingerFrame.x, fingerFrame.y, fingerFrame.width, fingerFrame.height);
      
      // Clear the coin frame area (make it transparent)
      ctx.clearRect(coinFrame.x, coinFrame.y, coinFrame.width, coinFrame.height);

      // Detect coin BEFORE drawing any overlays
      const coinPresent = detectCoin(videoRef.current, coinFrame);
      setCoinDetected(coinPresent);

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

      // Draw circle detection indicators on coin frame
      ctx.strokeStyle = coinPresent ? "#10b981" : "rgba(255, 255, 255, 0.5)";
      ctx.lineWidth = 3;
      ctx.strokeRect(
        coinFrame.x,
        coinFrame.y,
        coinFrame.width,
        coinFrame.height
      );

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
              const distance = checkFingerDistance(
                fingerSize,
                canvas.height,
                sizeHistory
              );
              const newD = checkDistance(
                fingerSize,
                canvas.height,
                sizeHistory
              );
              setDValue(newD)
              setDistanceStatus(distance);
              setSizeHistory((prev) => [
                ...prev.slice(-4),
                fingerSize / canvas.height,
              ]);
            //   setDValue(fingerSize / canvas.height);
              distanceOk = distance === "PERFECT";

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
  }, [
    handposeModel,
    capturedImage,
    sizeHistory,
    fingerFrame,
    coinFrame,
    isImagesLoaded,
    detectCoin,
  ]);

  const handleCapture = useCallback(() => {
    if (!fingerInFrame || !coinDetected || distanceStatus !== "PERFECT") {
      console.log("Capture blocked - Requirements not met:", {
        fingerInFrame,
        coinDetected,
        distanceStatus
      });
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

      // Log the image data that will be sent to backend
      console.log("=== IMAGE CAPTURED FOR BACKEND ===");
      console.log("Image Data (Base64):", imageData.substring(0, 100) + "...");
      console.log("Image Size:", imageData.length, "characters");
      console.log("Image Dimensions:", {
        width: capturedCanvas.width,
        height: capturedCanvas.height
      });
      console.log("Capture Conditions:", {
        fingerDetected: fingerInFrame,
        coinDetected: coinDetected,
        distance: distanceStatus
      });
      console.log("=== READY TO SEND TO BACKEND ===");

      // This is the data structure you would send to backend
      const backendPayload = {
        imageData: imageData,
        dimensions: {
          width: capturedCanvas.width,
          height: capturedCanvas.height
        },
        metadata: {
          fingerDetected: fingerInFrame,
          coinDetected: coinDetected,
          distanceStatus: distanceStatus,
          timestamp: new Date().toISOString()
        }
      };
      console.log("Backend Payload:", backendPayload);

      setCapturedImage(imageData);
      isDetectionActiveRef.current = false;
    } catch (err) {
      console.error("Capture error:", err);
    }
  }, [fingerInFrame, coinDetected, distanceStatus]);

  const resetCapture = useCallback(async () => {
    setCapturedImage(null);
    setDistanceStatus("");
    setDValue(0);
    setFingerInFrame(false);
    setCoinDetected(false);
    setSizeHistory([]);
    detectionHistoryRef.current = [];
    setError(null);

    await setupCamera();
    isDetectionActiveRef.current = true;
    
    turnOnFlash();
  }, [setupCamera]);

  useEffect(() => {
    setupCamera();
    loadModels();

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
    if (handposeModel && isOpenCVLoaded && !isLoading && !capturedImage && isImagesLoaded) {
      isDetectionActiveRef.current = true;
      detectHand();
    }

    return () => {
      isDetectionActiveRef.current = false;
      if (detectionLoopRef.current) {
        clearTimeout(detectionLoopRef.current);
      }
    };
  }, [handposeModel, isOpenCVLoaded, isLoading, capturedImage, detectHand, isImagesLoaded]);

  if (error) {
    return (
      <div style={styles.container}>
        <div style={styles.errorContainer}>
          <div style={styles.errorIcon}>⚠️</div>
          <p style={styles.errorText}>{error}</p>
          <button onClick={resetCapture} style={styles.retryBtn}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (capturedImage) {
    return (
      <div style={styles.container}>
        <img src={capturedImage} alt="Captured" style={styles.capturedImage} />
        <div style={styles.captureSuccess}>
          <span style={styles.successIcon}>✓</span>
          <span style={styles.successText}>Image Captured</span>
        </div>
        <button onClick={resetCapture} style={styles.recaptureBtn}>
          Capture Again
        </button>
      </div>
    );
  }

  const isReadyToCapture =
    fingerInFrame && coinDetected && distanceStatus === "PERFECT";

  return (
    <div style={styles.container}>
      <video ref={videoRef} style={styles.video} autoPlay playsInline muted />
      <canvas ref={canvasRef} style={styles.canvas} />
      <canvas ref={capturedCanvasRef} style={{ display: "none" }} />

      {isLoading && (
        <div style={styles.loading}>
          <div style={styles.loaderContainer}>
            <div style={styles.loader}></div>
            <div style={styles.progressBar}>
              <div
                style={{ ...styles.progressFill, width: `${loadingProgress}%` }}
              />
            </div>
          </div>
          <p style={styles.loadingText}>
            {loadingProgress < 60
              ? "Initializing Camera..."
              : loadingProgress < 100
              ? "Loading AI Models..."
              : "Ready!"}
          </p>
        </div>
      )}

      {!isLoading && (
        <>
          <div style={styles.statusBar}>
            <div
              style={{
                ...styles.statusItem,
                ...(fingerInFrame && styles.statusActive),
              }}
            >
              <span style={styles.statusIcon}>
                {fingerInFrame ? "✓" : "○"}
              </span>
              <span style={styles.statusText}>Finger Detected</span>
            </div>
            <div
              style={{
                ...styles.statusItem,
                ...(coinDetected && styles.statusActive),
              }}
            >
              <span style={styles.statusIcon}>
                {coinDetected ? "✓" : "○"}
              </span>
              <span style={styles.statusText}>Circle Detected</span>
            </div>
            <div
              style={{
                ...styles.statusItem,
                ...(distanceStatus === "PERFECT" && styles.statusActive),
              }}
            >
              <span style={styles.statusIcon}>
                {distanceStatus === "PERFECT" ? "✓" : "○"}
              </span>
              <span style={styles.statusText}>Distance Okay</span>
            </div>
            <div style={styles.distanceValue}>
              <span style={styles.distanceLabel}>D:</span>
              <span style={styles.distanceNumber}>{dValue?.toFixed(3)}</span>
            </div>
          </div>

          {distanceStatus && distanceStatus !== "PERFECT" && (
            <div
              style={{
                ...styles.distanceIndicator,
                top: `${fingerFrame.y - 60}px`,
                left: `${fingerFrame.x + fingerFrame.width / 2}px`,
                ...(distanceStatus === "TOO FAR" && styles.tooFar),
                ...(distanceStatus === "TOO CLOSE" && styles.tooClose),
                ...(distanceStatus.includes("FINGER") && styles.showFingers),
                ...(distanceStatus.includes("INDEX") && styles.showFingers),
              }}
            >
              <span style={styles.distanceText}>{distanceStatus}</span>
            </div>
          )}

          <button
            onClick={handleCapture}
            disabled={!isReadyToCapture}
            style={{
              ...styles.captureBtn,
              ...(isReadyToCapture
                ? styles.captureBtnReady
                : styles.captureBtnDisabled),
            }}
          >
            {isReadyToCapture ? "📸 Capture Now" : "Align Finger & Coin"}
          </button>
        </>
      )}
    </div>
  );
}

const styles = {
  container: {
    position: "fixed",
    top: 0,
    left: 0,
    width: "100vw",
    height: "100vh",
    backgroundColor: "black",
    overflow: "hidden",
  },
  video: {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    objectFit: "cover",
  },
  canvas: {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
  },
  capturedImage: {
    width: "100%",
    height: "100%",
    objectFit: "contain",
  },
  loading: {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    textAlign: "center",
    color: "white",
    zIndex: 100,
    padding: "0 5vw",
    width: "90vw",
    maxWidth: "400px",
  },
  loaderContainer: {
    marginBottom: "20px",
  },
  loader: {
    border: "6px solid rgba(255, 255, 255, 0.3)",
    borderTop: "6px solid white",
    borderRadius: "50%",
    width: "min(60px, 15vw)",
    height: "min(60px, 15vw)",
    animation: "spin 1s linear infinite",
    margin: "0 auto 15px",
  },
  progressBar: {
    width: "min(200px, 50vw)",
    height: "8px",
    backgroundColor: "rgba(255, 255, 255, 0.3)",
    borderRadius: "4px",
    overflow: "hidden",
    margin: "0 auto",
  },
  progressFill: {
    height: "100%",
    backgroundColor: "#10b981",
    transition: "width 0.3s ease",
  },
  loadingText: {
    fontSize: "clamp(1rem, 4vw, 1.2rem)",
    fontWeight: "bold",
    marginTop: "10px",
  },
  statusBar: {
    position: "absolute",
    top: "min(20px, 2vh)",
    right: "min(20px, 3vw)",
    display: "flex",
    flexDirection: "column",
    gap: "min(10px, 1.5vh)",
    zIndex: 10,
    maxWidth: "min(200px, 45vw)",
  },
  statusItem: {
    background: "rgba(0, 0, 0, 0.7)",
    padding: "min(10px, 2vw) min(18px, 3vw)",
    borderRadius: "25px",
    backdropFilter: "blur(10px)",
    display: "flex",
    alignItems: "center",
    gap: "min(10px, 2vw)",
    transition: "all 0.3s ease",
    minWidth: "min(180px, 40vw)",
  },
  statusActive: {
    background: "rgba(16, 185, 129, 0.9)",
    boxShadow: "0 0 20px rgba(16, 185, 129, 0.6)",
  },
  statusIcon: {
    fontSize: "clamp(0.9rem, 3vw, 1.1rem)",
    color: "white",
    fontWeight: "bold",
  },
  statusText: {
    color: "white",
    fontSize: "clamp(0.75rem, 2.5vw, 0.9rem)",
    fontWeight: "600",
    whiteSpace: "nowrap",
  },
  distanceValue: {
    background: "rgba(0, 0, 0, 0.7)",
    padding: "min(10px, 2vw) min(15px, 3vw)",
    borderRadius: "20px",
    backdropFilter: "blur(10px)",
    display: "flex",
    alignItems: "center",
    gap: "min(6px, 1.5vw)",
  },
  distanceLabel: {
    color: "rgba(255, 255, 255, 0.7)",
    fontSize: "clamp(0.75rem, 2.5vw, 0.85rem)",
    fontWeight: "500",
  },
  distanceNumber: {
    color: "#10b981",
    fontSize: "clamp(0.8rem, 2.8vw, 0.95rem)",
    fontWeight: "bold",
    fontFamily: "monospace",
  },
  captureBtn: {
    position: "absolute",
    bottom: "min(30px, 5vh)",
    left: "50%",
    transform: "translateX(-50%)",
    color: "white",
    fontSize: "clamp(1rem, 4vw, 1.2rem)",
    fontWeight: "bold",
    padding: "min(18px, 3vh) min(50px, 8vw)",
    border: "none",
    borderRadius: "50px",
    cursor: "pointer",
    zIndex: 20,
    transition: "all 0.3s ease",
    minWidth: "min(280px, 70vw)",
    maxWidth: "90vw",
    textAlign: "center",
  },
  captureBtnReady: {
    background: "linear-gradient(135deg, #10b981 0%, #059669 100%)",
    boxShadow: "0 10px 30px rgba(16, 185, 129, 0.5)",
  },
  captureBtnDisabled: {
    background: "rgba(107, 114, 128, 0.8)",
    boxShadow: "0 10px 30px rgba(107, 114, 128, 0.3)",
    cursor: "not-allowed",
    opacity: 0.7,
  },
  recaptureBtn: {
    position: "absolute",
    bottom: "min(30px, 5vh)",
    left: "50%",
    transform: "translateX(-50%)",
    background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    color: "white",
    fontSize: "clamp(1rem, 4vw, 1.2rem)",
    fontWeight: "bold",
    padding: "min(18px, 3vh) min(50px, 8vw)",
    border: "none",
    borderRadius: "50px",
    cursor: "pointer",
    zIndex: 20,
    boxShadow: "0 10px 30px rgba(102, 126, 234, 0.5)",
    transition: "all 0.3s ease",
    minWidth: "min(250px, 65vw)",
    maxWidth: "90vw",
    textAlign: "center",
  },
  captureSuccess: {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    background: "rgba(16, 185, 129, 0.95)",
    color: "white",
    padding: "min(20px, 3vh) min(40px, 6vw)",
    borderRadius: "15px",
    display: "flex",
    alignItems: "center",
    gap: "min(15px, 3vw)",
    boxShadow: "0 10px 30px rgba(16, 185, 129, 0.5)",
    maxWidth: "90vw",
  },
  successIcon: {
    fontSize: "clamp(2rem, 6vw, 2.5rem)",
  },
  successText: {
    fontSize: "clamp(1.1rem, 4vw, 1.3rem)",
    fontWeight: "bold",
  },
  errorContainer: {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    textAlign: "center",
    color: "white",
    zIndex: 100,
    padding: "0 5vw",
    width: "90vw",
    maxWidth: "400px",
  },
  errorIcon: {
    fontSize: "clamp(3rem, 10vw, 4rem)",
    marginBottom: "min(20px, 3vh)",
  },
  errorText: {
    fontSize: "clamp(1rem, 3.5vw, 1.2rem)",
    marginBottom: "min(30px, 4vh)",
    lineHeight: "1.5",
  },
  retryBtn: {
    background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    color: "white",
    fontSize: "clamp(0.95rem, 3.5vw, 1.1rem)",
    fontWeight: "bold",
    padding: "min(15px, 2.5vh) min(40px, 6vw)",
    border: "none",
    borderRadius: "50px",
    cursor: "pointer",
    boxShadow: "0 10px 30px rgba(102, 126, 234, 0.5)",
  },
  distanceIndicator: {
    position: "absolute",
    transform: "translate(-50%, 0)",
    zIndex: 10,
    padding: "min(12px, 2vh) min(20px, 3.5vw)",
    borderRadius: "12px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "min(10px, 2vw)",
    fontSize: "clamp(0.9rem, 3vw, 1rem)",
    fontWeight: "bold",
    backdropFilter: "blur(10px)",
    maxWidth: "85vw",
    textAlign: "center",
    whiteSpace: "nowrap",
  },
  tooFar: {
    background: "rgba(239, 68, 68, 0.9)",
    color: "white",
    boxShadow: "0 10px 30px rgba(239, 68, 68, 0.5)",
  },
  tooClose: {
    background: "rgba(251, 191, 36, 0.9)",
    color: "white",
    boxShadow: "0 10px 30px rgba(251, 191, 36, 0.5)",
  },
  showFingers: {
    background: "rgba(59, 130, 246, 0.9)",
    color: "white",
    boxShadow: "0 10px 30px rgba(59, 130, 246, 0.5)",
  },
  distanceText: {
    fontSize: "clamp(0.85rem, 3vw, 0.95rem)",
  },
};