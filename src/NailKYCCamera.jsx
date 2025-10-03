import * as handpose from "@tensorflow-models/handpose";
import * as tf from "@tensorflow/tfjs";
import React, { useEffect, useRef, useState } from "react";

const calculateHandSize = (hand) => {
  const bb = hand.boundingBox;
  const handHeight = bb.bottomRight[1] - bb.topLeft[1];
  const handWidth = bb.bottomRight[0] - bb.topLeft[0];
  return Math.max(handHeight, handWidth); // Max dimension for orientation-agnostic size
};

// Updated checkDistance (with new thresholds, smoothing, confidence)
const checkDistance = (handSize, canvasHeight, history) => {
  const relativeSize = handSize / canvasHeight;

  // Smoothing: Add to history (keep last 5)
  const newHistory = [...history.slice(-4), relativeSize];
  const avgSize =
    newHistory.reduce((sum, val) => sum + val, 0) / newHistory.length;

  console.log("avgRelativeSize ---->", avgSize, handSize, canvasHeight);

  if (avgSize < 0.75) {
    return "TOO FAR";
  } else if (avgSize > 1.25) {
    return "TOO CLOSE";
  } else {
    return "PERFECT";
  }
};

const drawNails = (ctx, landmarks) => {
  const fingertips = [4, 8, 12, 16, 20];
  const fingerNames = ["Thumb", "Index", "Middle", "Ring", "Pinky"];

  fingertips.forEach((tip, index) => {
    const [x, y] = landmarks[tip];

    ctx.strokeStyle = "#00ff00";
    // ctx.lineWidth = 4;
    ctx.beginPath();
    // ctx.arc(x, y, 20, 0, 2 * Math.PI);
    // ctx.stroke();

    ctx.fillStyle = "rgba(0, 255, 0, 0.2)";
    ctx.fill();

    ctx.fillStyle = "#00ff00";
    ctx.font = "bold 18px Arial";
    ctx.fillText(fingerNames[index], x - 30, y - 30);
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

    let isExtended = false;

    if (fingerId === 0) {
      isExtended = tip[0] < middle[0];
    } else {
      const isTipAboveMiddle = tip[1] < middle[1];
      const isMiddleAboveBase = middle[1] < base[1];
      isExtended = isTipAboveMiddle && isMiddleAboveBase;
    }

    const wrist = landmarks[0];
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

  const [model, setModel] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [distanceStatus, setDistanceStatus] = useState("");
  const [capturedImage, setCapturedImage] = useState(null);
  const [isFlashOn, setIsFlashOn] = useState(false);
  const [dValue, setDValue] = useState(0);
  const [sizeHistory, setSizeHistory] = useState([]); // Array for recent relative sizes

  // In detectHand (inside useEffect)
  const detectHand = async () => {
    try {
      if (videoRef.current.readyState === 4) {
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

          // Confidence filter: Ignore low-quality detections
          if (hand.handInViewConfidence < 0.8) {
            setDistanceStatus("");
            setSizeHistory([]); // Reset history on bad detection
            return;
          }

          const landmarks = hand.landmarks;
          const handSize = calculateHandSize(hand);
          const detectedFingers = detectFingers(landmarks); // Your existing function

          // Only set status if full hand visible (for NailKYC precision)
          if (detectedFingers.length === 5) {
            const distance = checkDistance(
              handSize,
              canvas.height,
              sizeHistory
            );
            setDistanceStatus(distance);
            setSizeHistory([
              ...sizeHistory.slice(-4),
              handSize / canvas.height,
            ]); // Update history
            setDValue(handSize / canvas.height);
          } else {
            setDistanceStatus("SHOW FULL HAND"); // New status for feedback
          }

          drawNails(ctx, landmarks);
        } else {
          setDistanceStatus("");
          setSizeHistory([]);
        }
      }
    } catch (err) {
      console.error("Detection error:", err);
    }

    // Throttle for mobile efficiency: Detect every 200ms instead of RAF
    setTimeout(detectHand, 200);
  };

  const turnOnFlash = async () => {
    if (isFlashOn) return; // Already on

    // Check if running in React Native WebView
    if (window.ReactNativeWebView && window.toggleNativeTorch) {
      window.toggleNativeTorch();
      setIsFlashOn(true);
    } else {
      // Fallback to browser API
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

  useEffect(() => {
    let animationFrame;

    const setupCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { exact: "environment" },

            // Mobile-optimized resolution for better performance
            width: { ideal: 640, max: 1280 },
            height: { ideal: 480, max: 720 },
            // frameRate: { ideal: 15, max: 20 },
            frameRate: { ideal: 30, max: 40 },
          },
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current.play();
          };

          // Check if flash/torch is supported (only if not in React Native)
          if (!window.ReactNativeWebView) {
            const track = stream.getVideoTracks()[0];
            const capabilities = track.getCapabilities();

            if (capabilities.torch) {
              // Flash support detected
            }
          }
        }
      } catch (err) {
        console.error("Camera error:", err);
      }
    };

    const loadModel = async () => {
      try {
        await tf.ready();
        // Set TensorFlow backend for mobile optimization
        await tf.setBackend("webgl");

        const handposeModel = await handpose.load();
        setModel(handposeModel);
        setIsLoading(false);

        // Turn on flashlight when model is loaded
        turnOnFlash();
      } catch (err) {
        console.error("Model error:", err);
      }
    };

    setupCamera();
    loadModel();

    // Listen for messages from React Native
    const handleMessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "TORCH_STATE") {
          setIsFlashOn(data.isOn);
        }
      } catch (error) {
        console.error("Message handling error:", error);
      }
    };

    if (window.ReactNativeWebView) {
      window.addEventListener("message", handleMessage);
      document.addEventListener("message", handleMessage); // For iOS
    }

    return () => {
      if (animationFrame) cancelAnimationFrame(animationFrame);
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      }
      if (window.ReactNativeWebView) {
        window.removeEventListener("message", handleMessage);
        document.removeEventListener("message", handleMessage);
      }
    };
  }, []);
  useEffect(() => {
    if (!model || !videoRef.current || isLoading) return;

    detectHand(); // Start throttled loop

    return () => {}; // No need for cancel as setTimeout is self-managing
  }, [model, isLoading, capturedImage]);

  const handleCapture = () => {
    if (distanceStatus !== "PERFECT") {
      alert("Please adjust your distance! Hand should be at perfect distance.");
      return;
    }

    // Capture the image from the canvas (which shows exactly what's on screen)
    const canvas = canvasRef.current;
    const capturedCanvas = capturedCanvasRef.current;
    const ctx = capturedCanvas.getContext("2d");

    // Set captured canvas to match the display canvas dimensions
    capturedCanvas.width = canvas.width;
    capturedCanvas.height = canvas.height;

    // Draw the video frame to match the canvas display
    const video = videoRef.current;
    ctx.drawImage(video, 0, 0, capturedCanvas.width, capturedCanvas.height);

    const imageData = capturedCanvas.toDataURL("image/png");

    // Console log the image data for backend
    console.log("=== CAPTURED IMAGE FOR BACKEND ===");
    console.log("Image Data URL:", imageData);
    console.log("Image Size:", imageData.length, "characters");
    console.log(
      "Image Dimensions:",
      capturedCanvas.width,
      "x",
      capturedCanvas.height
    );
    console.log("Video Dimensions:", video.videoWidth, "x", video.videoHeight);
    console.log("Canvas Dimensions:", canvas.width, "x", canvas.height);
    console.log("================================");

    setCapturedImage(imageData);
  };

  const resetCapture = () => {
    setCapturedImage(null);
    setDistanceStatus("");
    setDValue(0);

    // Restart the camera stream
    const setupCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { exact: "environment" },
            // Mobile-optimized resolution for better performance
            width: { ideal: 640, max: 1280 },
            height: { ideal: 480, max: 720 },
            frameRate: { ideal: 30, max: 40 },
          },
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current.play();
          };
        }
      } catch (err) {
        console.error("Camera restart error --->", err);
      }
    };

    setupCamera();

    // Turn flashlight back on when returning to capture mode
    turnOnFlash();
  };

  if (capturedImage) {
    return (
      <div style={styles.container}>
        <img src={capturedImage} alt="Captured" style={styles.capturedImage} />
        <button onClick={resetCapture} style={styles.recaptureBtn}>
          Capture Again
        </button>
      </div>
    );
  }

  const isReadyToCapture = distanceStatus === "PERFECT";

  return (
    <div style={styles.container}>
      <video ref={videoRef} style={styles.video} autoPlay playsInline muted />
      <canvas ref={canvasRef} style={styles.canvas} />
      <canvas ref={capturedCanvasRef} style={{ display: "none" }} />

      {isLoading && (
        <div style={styles.loading}>
          <div style={styles.loader}></div>
          <p style={styles.loadingText}>Loading AI Model...</p>
        </div>
      )}

      {!isLoading && (
        <>
          <div style={styles.fingerCounter}>
            <p style={styles.counterText}>D Value: {dValue?.toFixed(2)}</p>
          </div>

          {distanceStatus && (
            <div
              style={{
                ...styles.distanceIndicator,
                ...(distanceStatus === "TOO FAR" && styles.tooFar),
                ...(distanceStatus === "TOO CLOSE" && styles.tooClose),
                ...(distanceStatus === "PERFECT" && styles.perfect),
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
                  <span style={styles.distanceText}>Perfect Distance</span>
                </>
              )}
            </div>
          )}

          <button
            onClick={handleCapture}
            style={{
              ...styles.captureBtn,
              ...(isReadyToCapture
                ? styles.captureBtnReady
                : styles.captureBtnDisabled),
            }}
          >
            {isReadyToCapture ? "Capture" : "Adjust Position"}
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
  loader: {
    border: "6px solid rgba(255, 255, 255, 0.3)",
    borderTop: "6px solid white",
    borderRadius: "50%",
    width: "60px",
    height: "60px",
    animation: "spin 1s linear infinite",
    margin: "0 auto 15px",
  },
  loadingText: {
    fontSize: "1.2rem",
    fontWeight: "bold",
  },
  fingerCounter: {
    position: "absolute",
    top: "20px",
    left: "20px",
    textAlign: "center",
    zIndex: 10,
    background: "rgba(0, 0, 0, 0.7)",
    padding: "5px",
    borderRadius: "15px",
    backdropFilter: "blur(10px)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "column",
  },
  counterCircle: {
    width: "70px",
    height: "70px",
    borderRadius: "50%",
    background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "column",
    marginBottom: "8px",
    boxShadow: "0 10px 30px rgba(102, 126, 234, 0.5)",
  },
  counterNumber: {
    fontSize: "1.3rem",
    fontWeight: "bold",
    color: "white",
    lineHeight: 1,
  },
  counterLabel: {
    fontSize: "0.8rem",
    color: "rgba(255, 255, 255, 0.8)",
  },
  counterText: {
    color: "white",
    fontSize: "0.85rem",
    fontWeight: 600,
    margin: 0,
  },
  distanceIndicator: {
    position: "absolute",
    top: "20px",
    right: "20px",
    zIndex: 10,
    padding: "15px 25px",
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
    padding: "16px 50px",
    border: "none",
    borderRadius: "50px",
    cursor: "pointer",
    zIndex: 20,
    transition: "all 0.3s ease",
    minWidth: "200px",
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
    padding: "16px 50px",
    border: "none",
    borderRadius: "50px",
    cursor: "pointer",
    zIndex: 20,
    boxShadow: "0 10px 30px rgba(102, 126, 234, 0.5)",
    transition: "all 0.3s ease",
    minWidth: "200px",
    textAlign: "center",
  },
};