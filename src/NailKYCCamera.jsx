import * as handpose from "@tensorflow-models/handpose";
import * as tf from "@tensorflow/tfjs";
import React, { useCallback, useEffect, useRef, useState } from "react";

const calculateHandSize = (hand) => {
  const bb = hand.boundingBox;
  const handHeight = bb.bottomRight[1] - bb.topLeft[1];
  const handWidth = bb.bottomRight[0] - bb.topLeft[0];
  return Math.max(handHeight, handWidth);
};

const checkDistance = (handSize, canvasHeight, history) => {
  const relativeSize = handSize / canvasHeight;
  const newHistory = [...history.slice(-4), relativeSize];
  const avgSize = newHistory.reduce((sum, val) => sum + val, 0) / newHistory.length;

  const TOO_FAR_THRESHOLD = 0.75;    // Increase if "TOO FAR" triggers too easily
  const TOO_CLOSE_THRESHOLD = 1.25;  // Decrease if "TOO CLOSE" triggers too easily

  if (avgSize < TOO_FAR_THRESHOLD) return "TOO FAR";
  if (avgSize > TOO_CLOSE_THRESHOLD) return "TOO CLOSE";
  return "PERFECT";
};

const drawNails = (ctx, landmarks) => {
  const fingertips = [4, 8, 12, 16, 20];
  const fingerNames = ["Thumb", "Index", "Middle", "Ring", "Pinky"];

  fingertips.forEach((tip, index) => {
    const [x, y] = landmarks[tip];
    ctx.strokeStyle = "#00ff00";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(x, y, 15, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.fillStyle = "rgba(0, 255, 0, 0.3)";
    ctx.fill();
    ctx.fillStyle = "#00ff00";
    ctx.font = "bold 16px Arial";
    ctx.fillText(fingerNames[index], x - 25, y - 25);
  });
};

const detectFingers = (landmarks) => {
  const FINGER_LANDMARKS = [
    [4, 3, 2],
    [8, 6, 5],
    [12, 10, 9],
    [16, 14, 13],
    [20, 18, 17],
  ];

  const detectedFingers = [];

  FINGER_LANDMARKS.forEach(([tipIndex, middleIndex, baseIndex], fingerId) => {
    const tip = landmarks[tipIndex];
    const middle = landmarks[middleIndex];
    const base = landmarks[baseIndex];
    const wrist = landmarks[0];

    let isExtended = false;

    if (fingerId === 0) {
      isExtended = tip[0] < middle[0];
    } else {
      const isTipAboveMiddle = tip[1] < middle[1];
      const isMiddleAboveBase = middle[1] < base[1];
      isExtended = isTipAboveMiddle && isMiddleAboveBase;
    }

    const wristToTip = Math.sqrt(
      Math.pow(tip[0] - wrist[0], 2) + Math.pow(tip[1] - wrist[1], 2)
    );
    const isVisible = wristToTip > 80;

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
  const detectionLoopRef = useRef(null);
  const streamRef = useRef(null);
  const retryCountRef = useRef(0);

  const [model, setModel] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [distanceStatus, setDistanceStatus] = useState("");
  const [capturedImage, setCapturedImage] = useState(null);
  const [isFlashOn, setIsFlashOn] = useState(false);
  const [dValue, setDValue] = useState(0);
  const [detectedFingerCount, setDetectedFingerCount] = useState(0);
  const [sizeHistory, setSizeHistory] = useState([]);
  const [error, setError] = useState(null);
  const [isDetectionActive, setIsDetectionActive] = useState(false);

  // Send message to React Native
  const sendToNative = useCallback((type, data = {}) => {
    if (window.ReactNativeWebView) {
      window.ReactNativeWebView.postMessage(
        JSON.stringify({ type, ...data })
      );
    }
  }, []);

  // Toggle flash/torch
  const toggleFlash = useCallback(async (turnOn = true) => {
    if (window.ReactNativeWebView) {
      sendToNative("TOGGLE_TORCH", { enabled: turnOn });
      setIsFlashOn(turnOn);
    } else {
      try {
        const stream = streamRef.current;
        if (!stream) return;

        const track = stream.getVideoTracks()[0];
        const capabilities = track.getCapabilities();

        if (capabilities.torch) {
          await track.applyConstraints({
            advanced: [{ torch: turnOn }],
          });
          setIsFlashOn(turnOn);
        }
      } catch (err) {
        console.error("Flash toggle error:", err);
      }
    }
  }, [sendToNative]);

  // Setup camera with retry logic
  const setupCamera = useCallback(async (retryCount = 0) => {
    try {
      setLoadingProgress(20);
      
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { exact: "environment" },
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
            videoRef.current.play()
              .then(resolve)
              .catch(reject);
          };
          
          setTimeout(() => reject(new Error("Video load timeout")), 10000);
        });

        setLoadingProgress(60);
        sendToNative("CAMERA_READY");
        retryCountRef.current = 0;
      }
    } catch (err) {
      console.error("Camera setup error:", err);
      
      if (retryCount < 3) {
        setTimeout(() => setupCamera(retryCount + 1), 2000);
      } else {
        setError("Camera access failed. Please check permissions.");
        sendToNative("CAMERA_ERROR", { error: err.message });
      }
    }
  }, [sendToNative]);

  // Load TensorFlow model
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
      sendToNative("MODEL_LOADED");
      
      // Turn on flash after model loads
      setTimeout(() => toggleFlash(true), 500);
    } catch (err) {
      console.error("Model loading error:", err);
      setError("AI model failed to load. Please restart.");
      sendToNative("MODEL_ERROR", { error: err.message });
    }
  }, [sendToNative, toggleFlash]);

  // Hand detection loop with performance optimization
  const detectHand = useCallback(async () => {
    if (!model || !videoRef.current || !isDetectionActive || capturedImage) {
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
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (predictions.length > 0) {
        const hand = predictions[0];

        if (hand.handInViewConfidence < 0.75) {
          setDistanceStatus("");
          setDetectedFingerCount(0);
          setSizeHistory([]);
          detectionLoopRef.current = setTimeout(detectHand, 200);
          return;
        }

        const landmarks = hand.landmarks;
        const handSize = calculateHandSize(hand);
        const detectedFingers = detectFingers(landmarks);
        
        setDetectedFingerCount(detectedFingers.length);

        if (detectedFingers.length === 5) {
          const distance = checkDistance(handSize, canvas.height, sizeHistory);
          setDistanceStatus(distance);
          setSizeHistory(prev => [...prev.slice(-4), handSize / canvas.height]);
          setDValue(handSize / canvas.height);
          
          drawNails(ctx, landmarks);
          
          sendToNative("HAND_DETECTED", {
            fingerCount: 5,
            distance: distance,
            canCapture: distance === "PERFECT"
          });
        } else {
          setDistanceStatus("SHOW ALL FINGERS");
          sendToNative("HAND_DETECTED", {
            fingerCount: detectedFingers.length,
            distance: "INCOMPLETE",
            canCapture: false
          });
        }
      } else {
        setDistanceStatus("");
        setDetectedFingerCount(0);
        setSizeHistory([]);
      }
    } catch (err) {
      console.error("Detection error:", err);
    }

    detectionLoopRef.current = setTimeout(detectHand, 150);
  }, [model, isDetectionActive, capturedImage, sizeHistory, sendToNative]);

  // Capture image
  const handleCapture = useCallback(() => {
    if (distanceStatus !== "PERFECT") {
      sendToNative("CAPTURE_FAILED", { 
        reason: "Distance not perfect" 
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

      setCapturedImage(imageData);
      setIsDetectionActive(false);
      
      // Turn off flash after capture
      toggleFlash(false);

      sendToNative("IMAGE_CAPTURED", {
        imageData: imageData,
        dimensions: {
          width: capturedCanvas.width,
          height: capturedCanvas.height
        }
      });
    } catch (err) {
      console.error("Capture error:", err);
      sendToNative("CAPTURE_ERROR", { error: err.message });
    }
  }, [distanceStatus, sendToNative, toggleFlash]);

  // Reset and recapture
  const resetCapture = useCallback(async () => {
    setCapturedImage(null);
    setDistanceStatus("");
    setDValue(0);
    setDetectedFingerCount(0);
    setSizeHistory([]);
    setError(null);
    
    await setupCamera();
    setIsDetectionActive(true);
    
    setTimeout(() => toggleFlash(true), 500);
    sendToNative("RESET_CAPTURE");
  }, [setupCamera, toggleFlash, sendToNative]);

  // Listen to messages from React Native
  useEffect(() => {
    const handleMessage = (event) => {
      try {
        const data = typeof event.data === 'string' 
          ? JSON.parse(event.data) 
          : event.data;

        switch (data.type) {
          case "TORCH_STATE":
            setIsFlashOn(data.isOn);
            break;
          case "TRIGGER_CAPTURE":
            handleCapture();
            break;
          case "RESET_CAMERA":
            resetCapture();
            break;
          case "TOGGLE_DETECTION":
            setIsDetectionActive(data.enabled);
            break;
          default:
            break;
        }
      } catch (error) {
        console.error("Message handling error:", error);
      }
    };

    if (window.ReactNativeWebView) {
      window.addEventListener("message", handleMessage);
      document.addEventListener("message", handleMessage);
    }

    return () => {
      if (window.ReactNativeWebView) {
        window.removeEventListener("message", handleMessage);
        document.removeEventListener("message", handleMessage);
      }
    };
  }, [handleCapture, resetCapture]);

  // Initialize camera and model
  useEffect(() => {
    setupCamera();
    loadModel();

    return () => {
      if (detectionLoopRef.current) {
        clearTimeout(detectionLoopRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      toggleFlash(false);
    };
  }, []);

  // Start detection when model is ready
  useEffect(() => {
    if (model && !isLoading && !capturedImage) {
      setIsDetectionActive(true);
      detectHand();
    }

    return () => {
      setIsDetectionActive(false);
      if (detectionLoopRef.current) {
        clearTimeout(detectionLoopRef.current);
      }
    };
  }, [model, isLoading, capturedImage, detectHand]);

  // Error state
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

  // Captured image view
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

  const isReadyToCapture = distanceStatus === "PERFECT" && detectedFingerCount === 5;

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
                style={{
                  ...styles.progressFill,
                  width: `${loadingProgress}%`
                }}
              />
            </div>
          </div>
          <p style={styles.loadingText}>
            {loadingProgress < 60 ? "Initializing Camera..." : 
             loadingProgress < 100 ? "Loading AI Model..." : "Ready!"}
          </p>
        </div>
      )}

      {!isLoading && (
        <>
          <div style={styles.statusBar}>
            <div style={styles.leftPanel}>
              <div style={styles.fingerCounter}>
                <span style={styles.fingerIcon}>🖐️</span>
                <span style={styles.counterText}>
                  {detectedFingerCount}/5 Fingers
                </span>
              </div>
              <div style={styles.distanceValue}>
                <span style={styles.distanceLabel}>Distance:</span>
                <span style={styles.distanceNumber}>
                  {dValue.toFixed(3)}
                </span>
              </div>
            </div>
            
            {isFlashOn && (
              <div style={styles.flashIndicator}>
                <span style={styles.flashIcon}>💡</span>
              </div>
            )}
          </div>

          {distanceStatus && (
            <div
              style={{
                ...styles.distanceIndicator,
                ...(distanceStatus === "TOO FAR" && styles.tooFar),
                ...(distanceStatus === "TOO CLOSE" && styles.tooClose),
                ...(distanceStatus === "PERFECT" && styles.perfect),
                ...(distanceStatus === "SHOW ALL FINGERS" && styles.showFingers),
              }}
            >
              {distanceStatus === "TOO FAR" && (
                <>
                  <span style={styles.distanceIcon}>👋</span>
                  <span style={styles.distanceText}>Move Closer</span>
                </>
              )}
              {distanceStatus === "TOO CLOSE" && (
                <>
                  <span style={styles.distanceIcon}>✋</span>
                  <span style={styles.distanceText}>Move Back</span>
                </>
              )}
              {distanceStatus === "PERFECT" && (
                <>
                  <span style={styles.distanceIcon}>✓</span>
                  <span style={styles.distanceText}>Perfect!</span>
                </>
              )}
              {distanceStatus === "SHOW ALL FINGERS" && (
                <>
                  <span style={styles.distanceIcon}>🖐️</span>
                  <span style={styles.distanceText}>Show All 5 Fingers</span>
                </>
              )}
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
            {isReadyToCapture ? "📸 Capture" : "Position Your Hand"}
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
    padding: "0 20px",
  },
  loaderContainer: {
    marginBottom: "20px",
  },
  loader: {
    border: "6px solid rgba(255, 255, 255, 0.3)",
    borderTop: "6px solid white",
    borderRadius: "50%",
    width: "60px",
    height: "60px",
    animation: "spin 1s linear infinite",
    margin: "0 auto 15px",
  },
  progressBar: {
    width: "200px",
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
    fontSize: "1.2rem",
    fontWeight: "bold",
    marginTop: "10px",
  },
  statusBar: {
    position: "absolute",
    top: "20px",
    left: "20px",
    right: "20px",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-start",
    zIndex: 10,
  },
  leftPanel: {
    display: "flex",
    flexDirection: "column",
    gap: "10px",
  },
  fingerCounter: {
    background: "rgba(0, 0, 0, 0.7)",
    padding: "12px 20px",
    borderRadius: "25px",
    backdropFilter: "blur(10px)",
    display: "flex",
    alignItems: "center",
    gap: "10px",
  },
  fingerIcon: {
    fontSize: "1.5rem",
  },
  counterText: {
    color: "white",
    fontSize: "1rem",
    fontWeight: "600",
  },
  distanceValue: {
    background: "rgba(0, 0, 0, 0.7)",
    padding: "10px 18px",
    borderRadius: "20px",
    backdropFilter: "blur(10px)",
    display: "flex",
    alignItems: "center",
    gap: "8px",
  },
  distanceLabel: {
    color: "rgba(255, 255, 255, 0.7)",
    fontSize: "0.85rem",
    fontWeight: "500",
  },
  distanceNumber: {
    color: "#10b981",
    fontSize: "1rem",
    fontWeight: "bold",
    fontFamily: "monospace",
  },
  flashIndicator: {
    background: "rgba(251, 191, 36, 0.9)",
    padding: "10px",
    borderRadius: "50%",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    width: "45px",
    height: "45px",
  },
  flashIcon: {
    fontSize: "1.3rem",
  },
  distanceIndicator: {
    position: "absolute",
    top: "100px",
    left: "50%",
    transform: "translateX(-50%)",
    zIndex: 10,
    padding: "15px 30px",
    borderRadius: "12px",
    display: "flex",
    alignItems: "center",
    gap: "10px",
    fontSize: "1.2rem",
    fontWeight: "bold",
    backdropFilter: "blur(10px)",
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
  perfect: {
    background: "rgba(16, 185, 129, 0.9)",
    color: "white",
    boxShadow: "0 10px 30px rgba(16, 185, 129, 0.5)",
  },
  showFingers: {
    background: "rgba(59, 130, 246, 0.9)",
    color: "white",
    boxShadow: "0 10px 30px rgba(59, 130, 246, 0.5)",
  },
  distanceIcon: {
    fontSize: "2rem",
  },
  distanceText: {
    fontSize: "1.2rem",
  },
  captureBtn: {
    position: "absolute",
    bottom: "30px",
    left: "50%",
    transform: "translateX(-50%)",
    color: "white",
    fontSize: "1.2rem",
    fontWeight: "bold",
    padding: "18px 50px",
    border: "none",
    borderRadius: "50px",
    cursor: "pointer",
    zIndex: 20,
    transition: "all 0.3s ease",
    minWidth: "250px",
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
    bottom: "30px",
    left: "50%",
    transform: "translateX(-50%)",
    background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    color: "white",
    fontSize: "1.2rem",
    fontWeight: "bold",
    padding: "18px 50px",
    border: "none",
    borderRadius: "50px",
    cursor: "pointer",
    zIndex: 20,
    boxShadow: "0 10px 30px rgba(102, 126, 234, 0.5)",
    transition: "all 0.3s ease",
    minWidth: "250px",
    textAlign: "center",
  },
  captureSuccess: {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    background: "rgba(16, 185, 129, 0.95)",
    color: "white",
    padding: "20px 40px",
    borderRadius: "15px",
    display: "flex",
    alignItems: "center",
    gap: "15px",
    boxShadow: "0 10px 30px rgba(16, 185, 129, 0.5)",
  },
  successIcon: {
    fontSize: "2.5rem",
  },
  successText: {
    fontSize: "1.3rem",
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
    padding: "0 30px",
  },
  errorIcon: {
    fontSize: "4rem",
    marginBottom: "20px",
  },
  errorText: {
    fontSize: "1.2rem",
    marginBottom: "30px",
    lineHeight: "1.5",
  },
  retryBtn: {
    background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    color: "white",
    fontSize: "1.1rem",
    fontWeight: "bold",
    padding: "15px 40px",
    border: "none",
    borderRadius: "50px",
    cursor: "pointer",
    boxShadow: "0 10px 30px rgba(102, 126, 234, 0.5)",
  },
};