import * as handpose from "@tensorflow-models/handpose";
import * as tf from "@tensorflow/tfjs";
import React, { useCallback, useEffect, useRef, useState } from "react";

// ============================================
// 🎯 TUNABLE PARAMETERS - ADJUST THESE VALUES
// ============================================
const BEND_CONFIG = {
  // Angle thresholds (in degrees) - Increased for more tolerance
  ANGLE1_THRESHOLD: 170,    // Max angle at tip joint (higher = more tolerance)
  ANGLE2_THRESHOLD: 175,    // Max angle at middle joint (higher = more tolerance)
  
  // Distance thresholds
  DISTANCE_RATIO_MAX: 3.0,  // Increased max ratio for more flexibility
  
  // Z-depth thresholds for bend towards camera (smaller z = closer to camera)
  Z_DIFF_THRESHOLD: -10,    // Relaxed threshold for z diff (less negative = more tolerance)
  
  // Distance from camera
  TOO_FAR_THRESHOLD: 0.15,  // Finger too far [0.10-0.20]
  TOO_CLOSE_THRESHOLD: 0.55, // Finger too close [0.35-0.55]
  
  // Stability
  HISTORY_LENGTH: 5,        // Number of frames to average [3-8]
  STABLE_BEND_REQUIRED: 3,  // Minimum number of true bends in history for stable
  MIN_CONFIDENCE: 0.75,     // Min hand confidence [0.6-0.9]
  
  // New: Bend score threshold for isBent
  BEND_SCORE_THRESHOLD: 40, // Minimum bend score to consider bent (for tolerance)
};
// ============================================

const calculateFingerSize = (fingerTip, wrist) => {
  const distance = Math.sqrt(
    Math.pow(fingerTip[0] - wrist[0], 2) + Math.pow(fingerTip[1] - wrist[1], 2)
  );
  return distance;
};

const checkFingerDistance = (fingerSize, canvasHeight, history) => {
  const relativeSize = fingerSize / canvasHeight;
  const newHistory = [...history.slice(-BEND_CONFIG.HISTORY_LENGTH + 1), relativeSize];
  const avgSize =
    newHistory.reduce((sum, val) => sum + val, 0) / newHistory.length;

  if (avgSize < BEND_CONFIG.TOO_FAR_THRESHOLD) {
    return {
      status: "TOO FAR",
      dValue: avgSize
    };
  }
  if (avgSize > BEND_CONFIG.TOO_CLOSE_THRESHOLD) {
    return {
      status: "TOO CLOSE",
      dValue: avgSize
    };
  }
  return {
    status: "PERFECT",
    dValue: avgSize
  };
};

const calculateFingerBend = (landmarks) => {
  // Get index finger landmarks
  const tip = landmarks[8];      // Index fingertip
  const dip = landmarks[7];      // Distal interphalangeal joint
  const pip = landmarks[6];      // Proximal interphalangeal joint
  const mcp = landmarks[5];      // Metacarpophalangeal joint
  const wrist = landmarks[0];

  // Calculate vectors (using x,y only for angles)
  const v1 = [tip[0] - dip[0], tip[1] - dip[1]];
  const v2 = [dip[0] - pip[0], dip[1] - pip[1]];
  const v3 = [pip[0] - mcp[0], pip[1] - mcp[1]];

  // Calculate angles using dot product
  const dotProduct = (a, b) => a[0] * b[0] + a[1] * b[1];
  const magnitude = (v) => Math.sqrt(v[0] * v[0] + v[1] * v[1]);
  
  const angle1 = Math.acos(
    Math.max(-1, Math.min(1, dotProduct(v1, v2) / (magnitude(v1) * magnitude(v2))))
  ) * (180 / Math.PI);
  
  const angle2 = Math.acos(
    Math.max(-1, Math.min(1, dotProduct(v2, v3) / (magnitude(v2) * magnitude(v3))))
  ) * (180 / Math.PI);

  // Check if finger is pointing towards camera
  const tipToWristDist = Math.sqrt(
    Math.pow(tip[0] - wrist[0], 2) + Math.pow(tip[1] - wrist[1], 2)
  );
  const mcpToWristDist = Math.sqrt(
    Math.pow(mcp[0] - wrist[0], 2) + Math.pow(mcp[1] - wrist[1], 2)
  );

  const distanceRatio = tipToWristDist / mcpToWristDist;

  // Improved bend towards camera check using z-coordinates
  // z smaller (more negative) means closer to camera, relative to wrist
  const zBendCheck = tip[2] < dip[2] && dip[2] < pip[2]; // tip closer than dip closer than pip

  // Quantitative z diff for threshold
  const zDiff = tip[2] - pip[2];

  // Calculate bend score
  const bendScore = (180 - angle1) + (180 - angle2);

  return {
    angle1,
    angle2,
    distanceRatio,
    zBendCheck,
    zDiff,
    isBent: 
      angle1 < BEND_CONFIG.ANGLE1_THRESHOLD && 
      angle2 < BEND_CONFIG.ANGLE2_THRESHOLD  && 
      distanceRatio < BEND_CONFIG.DISTANCE_RATIO_MAX &&
      zDiff < BEND_CONFIG.Z_DIFF_THRESHOLD &&
      zBendCheck &&
      bendScore > BEND_CONFIG.BEND_SCORE_THRESHOLD,
    bendScore,
    tip,
    dip,
    pip,
    mcp,
    wrist,
  };
};

const areFingerLandmarksInZone = (bendData, zone) => {
  if (!bendData.tip || !bendData.dip || !bendData.pip) return false;
  
  const isInZone = (point) => {
    return (
      point[0] >= zone.x &&
      point[0] <= zone.x + zone.width &&
      point[1] >= zone.y &&
      point[1] <= zone.y + zone.height
    );
  };
  
  const points = [bendData.tip, bendData.dip, bendData.pip];
  const inZoneCount = points.filter(point => isInZone(point)).length;
  
  return inZoneCount >= 2; // At least 2 key points should be in zone
};

export default function NailDetect() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const capturedCanvasRef = useRef(null);
  const detectionLoopRef = useRef(null);
  const streamRef = useRef(null);
  const isDetectionActiveRef = useRef(false);

  const [handposeModel, setHandposeModel] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [capturedImage, setCapturedImage] = useState(null);
  const [statusMessage, setStatusMessage] = useState("Initializing...");
  const [isFingerBent, setIsFingerBent] = useState(false);
  const [fingerInZone, setFingerInZone] = useState(false);
  const [distanceStatus, setDistanceStatus] = useState("");
  const [error, setError] = useState(null);
  const [bendScore, setBendScore] = useState(0);
  const [dValue, setDValue] = useState(0);
  const [sizeHistory, setSizeHistory] = useState([]);
  const [bendHistory, setBendHistory] = useState([]);
  const [detectionZone, setDetectionZone] = useState({
    x: 0,
    y: 0,
    width: 800,
    height: 800,
  });
  const [angle1, setAngle1] = useState(0);
  const [angle2, setAngle2] = useState(0);
  const [zDiff, setZDiff] = useState(0);
  const [distanceRatio, setDistanceRatio] = useState(0);
  const [windowSize, setWindowSize] = useState({
    width: window.innerWidth,
    height: window.innerHeight
  });
  const [isFlashOn, setIsFlashOn] = useState(false)

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

  useEffect(() => {
    const handleResize = () => {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight
      });
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const setupCamera = useCallback(async (retryCount = 0) => {
    try {
      setLoadingProgress(20);

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
        //   facingMode: "user",
        facingMode: {exact: 'environment'},
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

      setLoadingProgress(100);
      setIsLoading(false);
      setStatusMessage("Ready!");

      setTimeout(() => {
        turnOnFlash();
      }, 500);
    } catch (err) {
      console.error("Model loading error:", err);
      setError("AI models failed to load. Please refresh.");
    }
  }, []);

  const detectHand = useCallback(async () => {
    if (
      !handposeModel ||
      !videoRef.current ||
      !isDetectionActiveRef.current ||
      capturedImage
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

      // Set canvas size to match video
      if (canvas.width !== videoRef.current.videoWidth) {
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;

        // Update detection zone to center, responsive size
        const scaleFactor = Math.min(
          windowSize.width / canvas.width,
          windowSize.height / canvas.height
        );
        const zoneSize = Math.min(canvas.width, canvas.height) * 0.5 * scaleFactor;
        const newZone = {
          x: (canvas.width - zoneSize) / 2,
          y: (canvas.height - zoneSize) / 2,
          width: zoneSize,
          height: zoneSize,
        };
        setDetectionZone(newZone);
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw dark overlay
      ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Clear detection zone
      ctx.clearRect(
        detectionZone.x,
        detectionZone.y,
        detectionZone.width,
        detectionZone.height
      );

      // Draw detection zone border
      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = 4;
      ctx.strokeRect(
        detectionZone.x,
        detectionZone.y,
        detectionZone.width,
        detectionZone.height
      );

      // Draw corner accents
      const cornerSize = Math.min(30, detectionZone.width * 0.1);
      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = 6;
      
      // Top-left
      ctx.beginPath();
      ctx.moveTo(detectionZone.x, detectionZone.y + cornerSize);
      ctx.lineTo(detectionZone.x, detectionZone.y);
      ctx.lineTo(detectionZone.x + cornerSize, detectionZone.y);
      ctx.stroke();

      // Top-right
      ctx.beginPath();
      ctx.moveTo(detectionZone.x + detectionZone.width - cornerSize, detectionZone.y);
      ctx.lineTo(detectionZone.x + detectionZone.width, detectionZone.y);
      ctx.lineTo(detectionZone.x + detectionZone.width, detectionZone.y + cornerSize);
      ctx.stroke();

      // Bottom-left
      ctx.beginPath();
      ctx.moveTo(detectionZone.x, detectionZone.y + detectionZone.height - cornerSize);
      ctx.lineTo(detectionZone.x, detectionZone.y + detectionZone.height);
      ctx.lineTo(detectionZone.x + cornerSize, detectionZone.y + detectionZone.height);
      ctx.stroke();

      // Bottom-right
      ctx.beginPath();
      ctx.moveTo(detectionZone.x + detectionZone.width - cornerSize, detectionZone.y + detectionZone.height);
      ctx.lineTo(detectionZone.x + detectionZone.width, detectionZone.y + detectionZone.height);
      ctx.lineTo(detectionZone.x + detectionZone.width, detectionZone.y + detectionZone.height - cornerSize);
      ctx.stroke();

      let bent = false;
      let stableBent = false;
      let inZone = false;
      let currentBendScore = 0;
      let distanceOk = false;

      if (predictions.length > 0) {
        const hand = predictions[0];

        if (hand.handInViewConfidence > BEND_CONFIG.MIN_CONFIDENCE) {
          const landmarks = hand.landmarks;
          
          // Calculate finger bend
          const bendData = calculateFingerBend(landmarks);
          
          // Check if finger keypoints are in zone FIRST
          inZone = areFingerLandmarksInZone(bendData, detectionZone);
          setFingerInZone(inZone);

          // Only process if finger is in zone
          if (inZone && bendData.tip && bendData.wrist) {
            // Check distance (too far/too close)
            const fingerSize = calculateFingerSize(bendData.tip, bendData.wrist);
            const { status: distance, dValue: newD } = checkFingerDistance(
              fingerSize,
              canvas.height,
              sizeHistory
            );
            
            setDValue(newD);
            setDistanceStatus(distance);
            setSizeHistory((prev) => [
              ...prev.slice(-BEND_CONFIG.HISTORY_LENGTH + 1),
              fingerSize / canvas.height,
            ]);
            
            distanceOk = distance === "PERFECT";
            
            // Check bend only if distance is OK
            if (distanceOk) {
              bent = bendData.isBent;
              currentBendScore = bendData.bendScore;

              // Update bend history
              setBendHistory((prev) => [
                ...prev.slice(-BEND_CONFIG.HISTORY_LENGTH + 1),
                bent ? 1 : 0,
              ]);

              // Calculate stable bent
              const bendCount = bendHistory.reduce((a, b) => a + b, 0);
              stableBent = bendCount >= BEND_CONFIG.STABLE_BEND_REQUIRED;

              setIsFingerBent(stableBent);
              setBendScore(currentBendScore);
              setAngle1(bendData.angle1);
              setAngle2(bendData.angle2);
              setZDiff(bendData.zDiff);
              setDistanceRatio(bendData.distanceRatio);
            } else {
              setIsFingerBent(false);
              setBendScore(0);
              setAngle1(0);
              setAngle2(0);
              setZDiff(0);
              setDistanceRatio(0);
              setBendHistory([]);
            }

            // Draw finger landmarks (only index finger - landmarks 5-8)
            const indexFingerIndices = [5, 6, 7, 8];
            
            // Draw connections between joints
            for (let i = 0; i < indexFingerIndices.length - 1; i++) {
              const idx1 = indexFingerIndices[i];
              const idx2 = indexFingerIndices[i + 1];
              const point1 = landmarks[idx1];
              const point2 = landmarks[idx2];
              
              ctx.beginPath();
              ctx.moveTo(point1[0], point1[1]);
              ctx.lineTo(point2[0], point2[1]);
              ctx.strokeStyle = stableBent && distanceOk ? "#10b981" : "#f59e0b";
              ctx.lineWidth = 3;
              ctx.stroke();
            }

            // Draw keypoint circles
            indexFingerIndices.forEach((idx) => {
              const point = landmarks[idx];
              ctx.beginPath();
              ctx.arc(point[0], point[1], 8, 0, 2 * Math.PI);
              ctx.fillStyle = stableBent && distanceOk ? "#10b981" : "#f59e0b";
              ctx.fill();
              ctx.strokeStyle = "white";
              ctx.lineWidth = 2;
              ctx.stroke();
            });

            // Update status message
            if (!distanceOk) {
              if (distance === "TOO FAR") {
                setStatusMessage("Move finger closer to camera");
              } else if (distance === "TOO CLOSE") {
                setStatusMessage("Move finger away from camera");
              }
            } else if (!stableBent) {
              setStatusMessage("Bend or lean finger towards camera to show nail curve");
            } else {
              setStatusMessage("Perfect! Nail curve visible - Ready to capture");
            }
          } else if (!inZone) {
            setStatusMessage("Move finger into detection zone");
            setIsFingerBent(false);
            setDistanceStatus("");
            setBendScore(0);
            setAngle1(0);
            setAngle2(0);
            setZDiff(0);
            setDistanceRatio(0);
            setBendHistory([]);
          }
        } else {
          setStatusMessage("Hand not detected clearly");
          setIsFingerBent(false);
          setFingerInZone(false);
          setDistanceStatus("");
          setBendScore(0);
          setAngle1(0);
          setAngle2(0);
          setZDiff(0);
          setDistanceRatio(0);
          setBendHistory([]);
        }
      } else {
        setStatusMessage("No hand detected");
        setIsFingerBent(false);
        setFingerInZone(false);
        setDistanceStatus("");
        setSizeHistory([]);
        setBendScore(0);
        setAngle1(0);
        setAngle2(0);
        setZDiff(0);
        setDistanceRatio(0);
        setBendHistory([]);
      }

    } catch (err) {
      console.error("Detection error:", err);
    }

    detectionLoopRef.current = setTimeout(detectHand, 100);
  }, [handposeModel, capturedImage, sizeHistory, detectionZone, bendHistory, windowSize]);

  const handleCapture = useCallback(() => {
    if (!isFingerBent || !fingerInZone || distanceStatus !== "PERFECT") {
      console.log("Capture blocked - Requirements not met:", {
        fingerInZone,
        isFingerBent,
        distanceStatus
      });
      return;
    }

    try {
      const capturedCanvas = capturedCanvasRef.current;
      const ctx = capturedCanvas.getContext("2d");
      const video = videoRef.current;

      // Crop to detection zone
      capturedCanvas.width = detectionZone.width;
      capturedCanvas.height = detectionZone.height;

      ctx.drawImage(
        video, 
        detectionZone.x, 
        detectionZone.y, 
        detectionZone.width, 
        detectionZone.height,
        0,
        0,
        detectionZone.width,
        detectionZone.height
      );

      const imageData = capturedCanvas.toDataURL("image/jpeg", 0.95);

      console.log("=== NAIL IMAGE CAPTURED FOR BACKEND ===");
      console.log("Image Data (Base64):", imageData);
      console.log("Image Size:", imageData.length, "characters");
      console.log("Dimensions:", {
        width: capturedCanvas.width,
        height: capturedCanvas.height
      });
      console.log("Bend Score:", bendScore);
      console.log("Distance Value:", dValue);

      setCapturedImage(imageData);
      isDetectionActiveRef.current = false;
    } catch (err) {
      console.error("Capture error:", err);
    }
  }, [isFingerBent, fingerInZone, distanceStatus, bendScore, dValue, detectionZone]);

  const resetCapture = useCallback(async () => {
    setCapturedImage(null);
    setStatusMessage("Initializing...");
    setIsFingerBent(false);
    setFingerInZone(false);
    setDistanceStatus("");
    setBendScore(0);
    setDValue(0);
    setSizeHistory([]);
    setBendHistory([]);
    setError(null);
    setAngle1(0);
    setAngle2(0);
    setZDiff(0);
    setDistanceRatio(0);

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
  }, [setupCamera, loadModels]);

  useEffect(() => {
    if (handposeModel && !isLoading && !capturedImage) {
      isDetectionActiveRef.current = true;
      detectHand();
    }

    return () => {
      isDetectionActiveRef.current = false;
      if (detectionLoopRef.current) {
        clearTimeout(detectionLoopRef.current);
      }
    };
  }, [handposeModel, isLoading, capturedImage, detectHand]);

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
        <button onClick={resetCapture} style={styles.recaptureBtn}>
          Capture Again
        </button>
      </div>
    );
  }

  const isReadyToCapture = isFingerBent && fingerInZone && distanceStatus === "PERFECT";

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
          {/* Status indicator - top left */}
          <div style={styles.statusContainer}>
            <div
              style={{
                ...styles.statusBadge,
                ...(isReadyToCapture ? styles.statusSuccess : styles.statusWarning),
              }}
            >
              <span style={styles.statusIcon}>
                {isReadyToCapture ? "✓" : "○"}
              </span>
              <span style={styles.statusText}>{statusMessage}</span>
            </div>
            
            {/* Debug info */}
            <div style={styles.debugInfo}>
              <div style={styles.debugItem}>
                <span style={styles.debugLabel}>In Zone:</span>
                <span style={{...styles.debugValue, color: fingerInZone ? '#10b981' : '#ef4444'}}>
                  {fingerInZone ? 'YES' : 'NO'}
                </span>
              </div>
              <div style={styles.debugItem}>
                <span style={styles.debugLabel}>Distance:</span>
                <span style={{
                  ...styles.debugValue, 
                  color: distanceStatus === 'PERFECT' ? '#10b981' : 
                         distanceStatus === 'TOO FAR' ? '#ef4444' :
                         distanceStatus === 'TOO CLOSE' ? '#f59e0b' : '#6b7280'
                }}>
                  {distanceStatus || 'N/A'}
                </span>
              </div>
              <div style={styles.debugItem}>
                <span style={styles.debugLabel}>Bent:</span>
                <span style={{...styles.debugValue, color: isFingerBent ? '#10b981' : '#ef4444'}}>
                  {isFingerBent ? 'YES' : 'NO'}
                </span>
              </div>
              <div style={styles.debugItem}>
                <span style={styles.debugLabel}>D-Value:</span>
                <span style={styles.debugValue}>{dValue.toFixed(3)}</span>
              </div>
              <div style={styles.debugItem}>
                <span style={styles.debugLabel}>Bend Score:</span>
                <span style={styles.debugValue}>{bendScore.toFixed(1)}</span>
              </div>
              <div style={styles.debugItem}>
                <span style={styles.debugLabel}>Angle1:</span>
                <span style={styles.debugValue}>{angle1.toFixed(1)}</span>
              </div>
              <div style={styles.debugItem}>
                <span style={styles.debugLabel}>Angle2:</span>
                <span style={styles.debugValue}>{angle2.toFixed(1)}</span>
              </div>
              <div style={styles.debugItem}>
                <span style={styles.debugLabel}>Z Diff:</span>
                <span style={styles.debugValue}>{zDiff.toFixed(1)}</span>
              </div>
              <div style={styles.debugItem}>
                <span style={styles.debugLabel}>Dist Ratio:</span>
                <span style={styles.debugValue}>{distanceRatio.toFixed(2)}</span>
              </div>
            </div>
          </div>

          {/* Distance indicator - appears above detection zone */}
          {distanceStatus && distanceStatus !== "PERFECT" && fingerInZone && (
            <div
              style={{
                ...styles.distanceIndicator,
                top: `${detectionZone.y - 60}px`,
                left: `${detectionZone.x + detectionZone.width / 2}px`,
                ...(distanceStatus === "TOO FAR" && styles.tooFar),
                ...(distanceStatus === "TOO CLOSE" && styles.tooClose),
              }}
            >
              <span style={styles.distanceText}>{distanceStatus}</span>
            </div>
          )}

          {/* Capture button - bottom center */}
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
            {isReadyToCapture ? "📸 Capture Nail" : "Align & Bend Finger"}
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
    fontSize: "1.1rem",
    fontWeight: "bold",
    marginTop: "10px",
  },
  statusContainer: {
    position: "absolute",
    top: "20px",
    left: "20px",
    zIndex: 10,
    display: "flex",
    flexDirection: "column",
    gap: "12px",
  },
  statusBadge: {
    padding: "12px 20px",
    borderRadius: "12px",
    backdropFilter: "blur(10px)",
    display: "flex",
    alignItems: "center",
    gap: "10px",
    maxWidth: "320px",
    boxShadow: "0 4px 20px rgba(0, 0, 0, 0.3)",
  },
  statusSuccess: {
    background: "rgba(16, 185, 129, 0.9)",
  },
  statusWarning: {
    background: "rgba(251, 191, 36, 0.9)",
  },
  statusIcon: {
    fontSize: "1.2rem",
    color: "white",
    fontWeight: "bold",
  },
  statusText: {
    color: "white",
    fontSize: "0.95rem",
    fontWeight: "600",
  },
  debugInfo: {
    background: "rgba(0, 0, 0, 0.8)",
    padding: "10px 15px",
    borderRadius: "10px",
    backdropFilter: "blur(10px)",
    display: "flex",
    flexDirection: "column",
    gap: "6px",
  },
  debugItem: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: "15px",
  },
  debugLabel: {
    color: "rgba(255, 255, 255, 0.7)",
    fontSize: "0.85rem",
    fontWeight: "500",
  },
  debugValue: {
    color: "#10b981",
    fontSize: "0.9rem",
    fontWeight: "bold",
    fontFamily: "monospace",
  },
  distanceIndicator: {
    position: "absolute",
    transform: "translate(-50%, 0)",
    zIndex: 10,
    padding: "12px 20px",
    borderRadius: "12px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "10px",
    fontSize: "1rem",
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
  distanceText: {
    fontSize: "0.95rem",
  },
  captureBtn: {
    position: "absolute",
    bottom: "30px",
    left: "50%",
    transform: "translateX(-50%)",
    color: "white",
    fontSize: "1.1rem",
    fontWeight: "bold",
    padding: "18px 50px",
    border: "none",
    borderRadius: "50px",
    cursor: "pointer",
    zIndex: 20,
    transition: "all 0.3s ease",
    minWidth: "280px",
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
    fontSize: "1.1rem",
    fontWeight: "bold",
    padding: "18px 50px",
    border: "none",
    borderRadius: "50px",
    cursor: "pointer",
    zIndex: 20,
    boxShadow: "0 10px 30px rgba(102, 126, 234, 0.5)",
    transition: "all 0.3s ease",
    minWidth: "250px",
  },
  errorContainer: {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    textAlign: "center",
    color: "white",
    zIndex: 100,
    padding: "0 20px",
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