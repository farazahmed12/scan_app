import * as handpose from "@tensorflow-models/handpose";
import * as tf from "@tensorflow/tfjs";
import React, { useCallback, useEffect, useRef, useState } from "react";

import fingerFrameImage from "./fingerFrame.png";

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

export default function NailKYCCamera() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const capturedCanvasRef = useRef(null);
  const detectionLoopRef = useRef(null);
  const streamRef = useRef(null);
  const isDetectionActiveRef = useRef(false);
  const coinFrameRef = useRef(null);
  const fingerFrameImageRef = useRef(null);

  const [model, setModel] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [distanceStatus, setDistanceStatus] = useState("");
  const [capturedImage, setCapturedImage] = useState(null);
  const [isFlashOn, setIsFlashOn] = useState(false);
  const [dValue, setDValue] = useState(0);
  const [fingerInFrame, setFingerInFrame] = useState(false);
  const [coinDetected, setCoinDetected] = useState(false);
  const [sizeHistory, setSizeHistory] = useState([]);
  const [error, setError] = useState(null);
  const [fingerFrame, setFingerFrame] = useState({
    x: 0,
    y: 0,
    width: 352, // Updated to match image dimensions
    height: 222,
  });
  const [coinFrame, setCoinFrame] = useState({
    x: 0,
    y: 0,
    width: 120,
    height: 120,
  });
  const [isFingerFrameLoaded, setIsFingerFrameLoaded] = useState(false);

  const sendToNative = useCallback((type, data = {}) => {
    if (window.ReactNativeWebView) {
      window.ReactNativeWebView.postMessage(JSON.stringify({ type, ...data }));
    }
  }, []);

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

  const setupCamera = useCallback(
    async (retryCount = 0) => {
      try {
        setLoadingProgress(20);

        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            // facingMode: { exact: "environment" },
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
          sendToNative("CAMERA_READY");
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
    },
    [sendToNative]
  );

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

      turnOnFlash();
    } catch (err) {
      console.error("Model loading error:", err);
      setError("AI model failed to load. Please restart.");
      sendToNative("MODEL_ERROR", { error: err.message });
    }
  }, [sendToNative]);

  // Preload finger frame image
  useEffect(() => {
    const img = new Image();
    img.src = fingerFrameImage;
    img.onload = () => {
      fingerFrameImageRef.current = img;
      setIsFingerFrameLoaded(true);
    };
    img.onerror = () => {
      console.error("Failed to load finger frame image");
    };
  }, []);

  // Update frame positions when canvas size changes
  useEffect(() => {
    const updateFrames = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      // Frame dimensions - using actual image dimensions
      const fingerFrameWidth = 352;
      const fingerFrameHeight = 222;
      const coinFrameSize = 120;
      const gap = 30; // Gap between frames

      // Calculate total width of both frames + gap
      const totalWidth = fingerFrameWidth + gap + coinFrameSize;

      // Center both frames horizontally
      const startX = (canvas.width - totalWidth) / 2;

      // Center vertically
      const centerY = canvas.height / 2;

      // Finger frame on the left
      setFingerFrame({
        x: startX,
        y: centerY - fingerFrameHeight / 2,
        width: fingerFrameWidth,
        height: fingerFrameHeight,
      });

      // Coin frame on the right
      setCoinFrame({
        x: startX + fingerFrameWidth + gap,
        y: centerY - coinFrameSize / 2,
        width: coinFrameSize,
        height: coinFrameSize,
      });
    };

    updateFrames();
  }, []);

  const detectCoinInFrame = (ctx, frame) => {
    try {
      // Get image data from the coin frame region
      const imageData = ctx.getImageData(
        frame.x,
        frame.y,
        frame.width,
        frame.height
      );
      const data = imageData.data;

      let coinColorPixels = 0;
      let edgePixels = 0;
      let brightPixels = 0;
      const totalPixels = frame.width * frame.height;

      // Analyze pixels for coin characteristics
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];

        // Calculate brightness
        const brightness = (r + g + b) / 3;

        // Detect metallic/reflective colors with stricter thresholds
        const isGoldenish =
          r > 180 && g > 140 && g < 200 && b < 120 && brightness > 150;
        const isSilverish =
          r > 180 &&
          g > 180 &&
          b > 180 &&
          Math.abs(r - g) < 30 &&
          Math.abs(g - b) < 30;
        const isCoppery = r > 180 && g > 100 && g < 160 && b < 100;

        if (isGoldenish || isSilverish || isCoppery) {
          coinColorPixels++;
        }

        // Count bright/reflective pixels
        if (brightness > 180) {
          brightPixels++;
        }
      }

      // Edge detection - coins have circular edges
      for (let y = 1; y < frame.height - 1; y++) {
        for (let x = 1; x < frame.width - 1; x++) {
          const idx = (y * frame.width + x) * 4;

          const centerBrightness =
            (data[idx] + data[idx + 1] + data[idx + 2]) / 3;

          // Check neighbors
          const topIdx = ((y - 1) * frame.width + x) * 4;
          const bottomIdx = ((y + 1) * frame.width + x) * 4;
          const leftIdx = (y * frame.width + (x - 1)) * 4;
          const rightIdx = (y * frame.width + (x + 1)) * 4;

          const topBrightness =
            (data[topIdx] + data[topIdx + 1] + data[topIdx + 2]) / 3;
          const bottomBrightness =
            (data[bottomIdx] + data[bottomIdx + 1] + data[bottomIdx + 2]) / 3;
          const leftBrightness =
            (data[leftIdx] + data[leftIdx + 1] + data[leftIdx + 2]) / 3;
          const rightBrightness =
            (data[rightIdx] + data[rightIdx + 1] + data[rightIdx + 2]) / 3;

          const edgeStrength =
            Math.abs(centerBrightness - topBrightness) +
            Math.abs(centerBrightness - bottomBrightness) +
            Math.abs(centerBrightness - leftBrightness) +
            Math.abs(centerBrightness - rightBrightness);

          if (edgeStrength > 100) {
            edgePixels++;
          }
        }
      }

      const coinRatio = coinColorPixels / totalPixels;
      const brightRatio = brightPixels / totalPixels;
      const edgeRatio = edgePixels / totalPixels;

      // Coin must meet multiple criteria:
      // 1. Sufficient metallic color pixels (at least 15%)
      // 2. Sufficient bright/reflective pixels (at least 20%)
      // 3. Visible edges indicating a circular object (at least 5%)
      const hasCoinColors = coinRatio > 0.15;
      const hasBrightness = brightRatio > 0.2;
      const hasEdges = edgeRatio > 0.05;

      // All three conditions must be met
      return hasCoinColors && hasBrightness && hasEdges;
    } catch (err) {
      console.error("Coin detection error:", err);
      return false;
    }
  };

  const detectHand = useCallback(async () => {
    if (
      !model ||
      !videoRef.current ||
      !isDetectionActiveRef.current ||
      capturedImage ||
      !isFingerFrameLoaded
    ) {
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
        const coinFrameSize = 120;
        const gap = 30;

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

      // Draw finger frame image instead of rectangle
      if (fingerFrameImageRef.current) {
        ctx.drawImage(
          fingerFrameImageRef.current,
          fingerFrame.x,
          fingerFrame.y,
          fingerFrame.width,
          fingerFrame.height
        );
      }

      // Draw green boundary around finger frame
      ctx.strokeStyle = fingerInFrame ? "#10b981" : "rgba(16, 185, 129, 0.5)";
      ctx.lineWidth = 3;
      ctx.strokeRect(
        fingerFrame.x,
        fingerFrame.y,
        fingerFrame.width,
        fingerFrame.height
      );

      // Draw coin frame (top-right)
      ctx.strokeStyle = coinDetected ? "#10b981" : "rgba(255, 255, 255, 0.5)";
      ctx.lineWidth = 3;
      ctx.strokeRect(
        coinFrame.x,
        coinFrame.y,
        coinFrame.width,
        coinFrame.height
      );
      ctx.fillStyle = coinDetected
        ? "rgba(16, 185, 129, 0.2)"
        : "rgba(255, 255, 255, 0.1)";
      ctx.fillRect(coinFrame.x, coinFrame.y, coinFrame.width, coinFrame.height);

      // Add labels
      ctx.fillStyle = "white";
      ctx.font = "bold 16px Arial";
      ctx.textAlign = "center";
      ctx.fillText(
        "Place Coin Here",
        coinFrame.x + coinFrame.width / 2,
        coinFrame.y - 10
      );

      // Detect coin in frame
      const coinPresent = detectCoinInFrame(ctx, coinFrame);
      setCoinDetected(coinPresent);

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

            // Check if finger tip is inside the frame
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
              setDistanceStatus(distance);
              setSizeHistory((prev) => [
                ...prev.slice(-4),
                fingerSize / canvas.height,
              ]);
              setDValue(fingerSize / canvas.height);
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

      const canCapture = fingerValid && distanceOk && coinPresent;

      sendToNative("DETECTION_STATUS", {
        fingerInFrame: fingerValid,
        coinDetected: coinPresent,
        distanceStatus: distanceStatus,
        canCapture: canCapture,
      });
    } catch (err) {
      console.error("Detection error:", err);
    }

    detectionLoopRef.current = setTimeout(detectHand, 150);
  }, [
    model,
    capturedImage,
    sizeHistory,
    fingerFrame,
    coinFrame,
    fingerInFrame,
    coinDetected,
    distanceStatus,
    sendToNative,
    isFingerFrameLoaded,
  ]);

  const handleCapture = useCallback(() => {
    if (!fingerInFrame || !coinDetected || distanceStatus !== "PERFECT") {
      sendToNative("CAPTURE_FAILED", {
        reason: `Requirements not met - Finger in frame: ${fingerInFrame}, Coin detected: ${coinDetected}, Distance: ${distanceStatus}`,
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
      isDetectionActiveRef.current = false;

      sendToNative("IMAGE_CAPTURED", {
        imageData: imageData,
        dimensions: {
          width: capturedCanvas.width,
          height: capturedCanvas.height,
        },
      });
    } catch (err) {
      console.error("Capture error:", err);
      sendToNative("CAPTURE_ERROR", { error: err.message });
    }
  }, [fingerInFrame, coinDetected, distanceStatus, sendToNative]);

  const resetCapture = useCallback(async () => {
    setCapturedImage(null);
    setDistanceStatus("");
    setDValue(0);
    setFingerInFrame(false);
    setCoinDetected(false);
    setSizeHistory([]);
    setError(null);

    await setupCamera();
    isDetectionActiveRef.current = true;

    turnOnFlash();
    sendToNative("RESET_CAPTURE");
  }, [setupCamera, sendToNative]);

  useEffect(() => {
    const handleMessage = (event) => {
      try {
        const data =
          typeof event.data === "string" ? JSON.parse(event.data) : event.data;

        switch (data.type) {
          case "TRIGGER_CAPTURE":
            handleCapture();
            break;
          case "RESET_CAMERA":
            resetCapture();
            break;
          case "TOGGLE_DETECTION":
            isDetectionActiveRef.current = data.enabled;
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
    if (model && !isLoading && !capturedImage && isFingerFrameLoaded) {
      isDetectionActiveRef.current = true;
      detectHand();
    }

    return () => {
      isDetectionActiveRef.current = false;
      if (detectionLoopRef.current) {
        clearTimeout(detectionLoopRef.current);
      }
    };
  }, [model, isLoading, capturedImage, detectHand, isFingerFrameLoaded]);

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
              ? "Loading AI Model..."
              : "Ready!"}
          </p>
        </div>
      )}

      {!isLoading && (
        <>
          <div style={styles.statusBar}>
            <div style={styles.statusGrid}>
              <div
                style={{
                  ...styles.statusItem,
                  ...(fingerInFrame && styles.statusActive),
                }}
              >
                <span style={styles.statusIcon}>
                  {fingerInFrame ? "✓" : "○"}
                </span>
                <span style={styles.statusText}>Finger in Frame</span>
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
                <span style={styles.statusText}>Coin Detected</span>
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
                <span style={styles.statusText}>Distance OK</span>
              </div>
            </div>
            <div style={styles.distanceValue}>
              <span style={styles.distanceLabel}>D:</span>
              <span style={styles.distanceNumber}>{dValue.toFixed(3)}</span>
            </div>
          </div>

          {distanceStatus && distanceStatus !== "PERFECT" && (
            <div
              style={{
                ...styles.distanceIndicator,
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
    gap: "10px",
  },
  statusGrid: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
    flex: 1,
  },
  statusItem: {
    background: "rgba(0, 0, 0, 0.7)",
    padding: "8px 15px",
    borderRadius: "20px",
    backdropFilter: "blur(10px)",
    display: "flex",
    alignItems: "center",
    gap: "8px",
    transition: "all 0.3s ease",
  },
  statusActive: {
    background: "rgba(16, 185, 129, 0.8)",
    boxShadow: "0 0 15px rgba(16, 185, 129, 0.5)",
  },
  statusIcon: {
    fontSize: "1rem",
    color: "white",
    fontWeight: "bold",
  },
  statusText: {
    color: "white",
    fontSize: "0.85rem",
    fontWeight: "600",
  },
  distanceValue: {
    background: "rgba(0, 0, 0, 0.7)",
    padding: "10px 15px",
    borderRadius: "20px",
    backdropFilter: "blur(10px)",
    display: "flex",
    alignItems: "center",
    gap: "6px",
  },
  distanceLabel: {
    color: "rgba(255, 255, 255, 0.7)",
    fontSize: "0.85rem",
    fontWeight: "500",
  },
  distanceNumber: {
    color: "#10b981",
    fontSize: "0.95rem",
    fontWeight: "bold",
    fontFamily: "monospace",
  },
  distanceIndicator: {
    position: "absolute",
    top: "45%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    zIndex: 10,
    padding: "15px 25px",
    borderRadius: "12px",
    display: "flex",
    alignItems: "center",
    gap: "10px",
    fontSize: "1.1rem",
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
  showFingers: {
    background: "rgba(59, 130, 246, 0.9)",
    color: "white",
    boxShadow: "0 10px 30px rgba(59, 130, 246, 0.5)",
  },
  distanceText: {
    fontSize: "0.95rem",
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