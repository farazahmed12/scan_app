import React, { useEffect, useRef, useState } from "react";
import coin from '../coin.png';

// CSS is embedded here for a single-file component
const styles = `
  .app-root {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    background-color: #1a1a1a;
    color: #f0f0f0;
    padding: 20px;
    box-sizing: border-box;
  }
  h1 {
    margin-bottom: 20px;
    color: #e0e0e0;
  }
  .video-container {
    position: relative;
    width: 640px;
    height: 480px;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    background-color: #000;
  }
  .video-feed {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  .overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
  }
  .center-frame {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 200px;
    height: 200px;
    transform: translate(-50%, -50%);
    border: 3px dashed rgba(0, 255, 127, 0.7);
    box-shadow: 0 0 15px rgba(0, 255, 127, 0.5);
    pointer-events: none;
    border-radius: 8px;
  }
  .controls {
    margin-top: 20px;
  }
  button {
    background-color: #007bff;
    color: white;
    border: none;
    padding: 12px 24px;
    font-size: 16px;
    border-radius: 5px;
    cursor: pointer;
    transition: background-color 0.3s, transform 0.1s;
  }
  button:hover {
    background-color: #0056b3;
  }
  button:active {
    transform: scale(0.98);
  }
  button:disabled {
    background-color: #555;
    cursor: not-allowed;
  }
  .status {
    margin-top: 20px;
    padding: 10px 15px;
    background-color: #2a2a2a;
    border-radius: 5px;
    min-width: 250px;
    text-align: center;
  }
  .error {
    color: #ff4d4d;
    background-color: rgba(255, 77, 77, 0.1);
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 15px;
  }
  .loading {
    color: #4da6ff;
    background-color: rgba(77, 166, 255, 0.1);
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 15px;
  }
`;

export default function CircleDetect() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isWebcamStarted, setIsWebcamStarted] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [permissionDenied, setPermissionDenied] = useState(false);
  const [isOpenCVLoaded, setIsOpenCVLoaded] = useState(false);
  const [isCircleStable, setIsCircleStable] = useState(false);

  // Using a ref for the animation loop ID
  const requestRef = useRef();
  
  // Refs for stabilization
  const detectionHistoryRef = useRef([]);
  const stableCircleRef = useRef(null);
  const HISTORY_LENGTH = 8; // Number of frames to consider
  const STABILITY_THRESHOLD = 6; // Minimum detections needed in history

  // Define the size of our detection zone
  const ZONE_SIZE = 200;
  // Canvas/video dimensions
  const CANVAS_WIDTH = 640;
  const CANVAS_HEIGHT = 480;

  // Stabilization: Track detection history to avoid flickering
  const updateDetectionHistory = (detectedCircles) => {
    // Add current detection state to history
    detectionHistoryRef.current.push(detectedCircles.length > 0);
    
    // Keep only recent history
    if (detectionHistoryRef.current.length > HISTORY_LENGTH) {
      detectionHistoryRef.current.shift();
    }
    
    // Count positive detections in history
    const positiveDetections = detectionHistoryRef.current.filter(d => d).length;
    
    // Update stable state only if threshold is met
    const shouldBeStable = positiveDetections >= STABILITY_THRESHOLD;
    
    if (shouldBeStable !== isCircleStable) {
      setIsCircleStable(shouldBeStable);
      
      // Store the stable circle for drawing
      if (shouldBeStable && detectedCircles.length > 0) {
        stableCircleRef.current = detectedCircles[0];
      } else if (!shouldBeStable) {
        stableCircleRef.current = null;
      }
    }
  };

  // Validate if detected circle is actually circular (not an irregular shape)
  const validateCircle = (grayImg, cx, cy, radius, cv) => {
    try {
      // Check if circle center and area are within bounds
      if (cx < radius || cy < radius || 
          cx + radius >= grayImg.cols || cy + radius >= grayImg.rows) {
        return false;
      }

      // Sample points around the circle perimeter
      const numSamples = 16;
      let validPoints = 0;
      let edgeStrengthSum = 0;
      
      for (let i = 0; i < numSamples; i++) {
        const angle = (2 * Math.PI * i) / numSamples;
        const px = Math.round(cx + radius * Math.cos(angle));
        const py = Math.round(cy + radius * Math.sin(angle));
        
        // Inner point (slightly inside the circle)
        const ix = Math.round(cx + (radius - 8) * Math.cos(angle));
        const iy = Math.round(cy + (radius - 8) * Math.sin(angle));
        
        // Outer point (slightly outside the circle)
        const ox = Math.round(cx + (radius + 8) * Math.cos(angle));
        const oy = Math.round(cy + (radius + 8) * Math.sin(angle));
        
        // Check if all points are within bounds
        if (px >= 0 && px < grayImg.cols && py >= 0 && py < grayImg.rows &&
            ix >= 0 && ix < grayImg.cols && iy >= 0 && iy < grayImg.rows &&
            ox >= 0 && ox < grayImg.cols && oy >= 0 && oy < grayImg.rows) {
          
          const perimeterPixel = grayImg.ucharPtr(py, px)[0];
          const innerPixel = grayImg.ucharPtr(iy, ix)[0];
          const outerPixel = grayImg.ucharPtr(oy, ox)[0];
          
          // Calculate gradient across the circle boundary
          const innerGradient = Math.abs(perimeterPixel - innerPixel);
          const outerGradient = Math.abs(perimeterPixel - outerPixel);
          const totalGradient = innerGradient + outerGradient;
          
          edgeStrengthSum += totalGradient;
          
          // Check for strong edge (significant difference on both sides)
          if (totalGradient > 40) {
            validPoints++;
          }
        }
      }
      
      // At least 65% of sampled points should show strong edge characteristics
      const circleScore = validPoints / numSamples;
      const avgEdgeStrength = edgeStrengthSum / numSamples;
      
      // Must pass both tests: good score AND strong average edge
      return circleScore >= 0.65 && avgEdgeStrength >= 30;
    } catch (err) {
      return false; // If validation fails, reject the circle
    }
  };

  // Draw the detected circles on the overlay canvas
  const drawPredictions = (objects) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    objects.forEach(circle => {
      const { x, y, radius } = circle;

      // Draw the circle outline
      ctx.strokeStyle = "lime";
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.stroke();
      
      // Draw the center point
      ctx.fillStyle = "red";
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fill();
    });
  };

  // Load OpenCV from a CDN
  useEffect(() => {
    // Check if OpenCV is already loaded
    if (typeof window.cv !== 'undefined' && window.cv.Mat) {
      console.log("OpenCV already loaded.");
      setIsOpenCVLoaded(true);
      return;
    }

    // Check if script is already being loaded
    const existingScript = document.querySelector('script[src*="opencv"]');
    if (existingScript) {
      console.log("OpenCV script already added, waiting for initialization...");
      const checkCV = setInterval(() => {
        if (typeof window.cv !== 'undefined' && window.cv.Mat) {
          console.log("OpenCV is ready.");
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
    script.id = 'opencv-script';
    
    script.onload = () => {
      // OpenCV.js needs time to initialize after script loads
      const checkCV = setInterval(() => {
        if (typeof window.cv !== 'undefined' && window.cv.Mat) {
          console.log("OpenCV is ready.");
          setIsOpenCVLoaded(true);
          clearInterval(checkCV);
        }
      }, 100);
      
      // Timeout after 10 seconds
      setTimeout(() => {
        clearInterval(checkCV);
        if (!isOpenCVLoaded) {
          console.error("OpenCV initialization timeout");
        }
      }, 10000);
    };
    
    script.onerror = (error) => {
      console.error("Failed to load OpenCV.js script:", error);
      console.log("Trying alternative CDN...");
      
      // Try alternative CDN
      const altScript = document.createElement('script');
      altScript.src = 'https://cdn.jsdelivr.net/npm/opencv.js@1.2.1/opencv.js';
      altScript.async = true;
      altScript.id = 'opencv-script-alt';
      
      altScript.onload = () => {
        const checkCV = setInterval(() => {
          if (typeof window.cv !== 'undefined' && window.cv.Mat) {
            console.log("OpenCV loaded from alternative CDN.");
            setIsOpenCVLoaded(true);
            clearInterval(checkCV);
          }
        }, 100);
        
        setTimeout(() => clearInterval(checkCV), 10000);
      };
      
      altScript.onerror = () => {
        console.error("All CDN sources failed. Please check your internet connection.");
      };
      
      document.body.appendChild(altScript);
    };
    
    document.body.appendChild(script);

    // Cleanup is minimal - we keep the script loaded
    return () => {
      // Don't remove the script on unmount to avoid reloading issues
    };
  }, []);

  // Main processing loop using requestAnimationFrame for real-time performance
  const processVideo = () => {
    if (!isWebcamStarted || !isOpenCVLoaded || !videoRef.current || !canvasRef.current || typeof window.cv === 'undefined') {
      return;
    }

    const video = videoRef.current;
    if (video.readyState !== 4) {
      // Wait until video is ready
      requestRef.current = requestAnimationFrame(processVideo);
      return;
    }

    try {
      const cv = window.cv;
      const src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
      const cap = new cv.VideoCapture(video);
      cap.read(src); // Read a frame from the video

      // Define the Region of Interest (ROI) in the center
      const zoneLeft = Math.floor((src.cols - ZONE_SIZE) / 2);
      const zoneTop = Math.floor((src.rows - ZONE_SIZE) / 2);
      const roiRect = new cv.Rect(zoneLeft, zoneTop, ZONE_SIZE, ZONE_SIZE);
      const roi = src.roi(roiRect);

      const gray = new cv.Mat();
      cv.cvtColor(roi, gray, cv.COLOR_RGBA2GRAY);
      
      // Apply median blur to reduce noise while preserving edges
      cv.medianBlur(gray, gray, 5);
      
      const circles = new cv.Mat();
      // Balanced parameters - sweet spot between accuracy and detection
      // dp=1.2: good balance
      // minDist=45: moderate separation
      // param1=90: medium-high Canny threshold
      // param2=40: moderate accumulator - filters some but not too strict
      // minRadius=18, maxRadius=80: good range for objects, excludes small features
      cv.HoughCircles(gray, circles, cv.HOUGH_GRADIENT, 1.2, 45, 90, 40, 18, 80);

      const detectedCircles = [];
      if (circles.cols > 0) {
        for (let i = 0; i < circles.cols; ++i) {
          const cx = circles.data32F[i * 3];
          const cy = circles.data32F[i * 3 + 1];
          const radius = circles.data32F[i * 3 + 2];
          
          // Validate circle quality
          if (validateCircle(gray, cx, cy, radius, cv)) {
            const x = cx + zoneLeft;
            const y = cy + zoneTop;
            detectedCircles.push({ x, y, radius });
          }
        }
      }
      
      // Update detection history for stabilization
      updateDetectionHistory(detectedCircles);
      
      setPredictions(detectedCircles);
      drawPredictions(detectedCircles);

      // Clean up memory
      src.delete();
      roi.delete();
      gray.delete();
      circles.delete();
    } catch (err) {
      console.error("Error processing video:", err);
    }

    // Loop
    requestRef.current = requestAnimationFrame(processVideo);
  };

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: CANVAS_WIDTH },
          height: { ideal: CANVAS_HEIGHT },
        },
        audio: false,
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setIsWebcamStarted(true);
      setPermissionDenied(false);
    } catch (err) {
      console.error("Error accessing webcam:", err);
      setPermissionDenied(true);
      setIsWebcamStarted(false);
    }
  };

  const stopWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsWebcamStarted(false);
    setPredictions([]);
    setIsCircleStable(false);
    detectionHistoryRef.current = [];
    stableCircleRef.current = null;
    // Clear canvas when stopping
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  };
  
  // Effect to start/stop the processing loop
  useEffect(() => {
    if (isWebcamStarted && isOpenCVLoaded) {
      requestRef.current = requestAnimationFrame(processVideo);
    } else {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
    }

    // Cleanup function to stop animation frame on component unmount
    return () => {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, [isWebcamStarted, isOpenCVLoaded]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, []);

  return (
    <>
      <style>{styles}</style>
      <div className="app-root">
        <h1>Real-Time Circle Detection</h1>
        
        {!isOpenCVLoaded && (
          <div className="loading">Loading OpenCV.js... Please wait.</div>
        )}
        
        {permissionDenied && (
          <div className="error">
            Camera permission was denied. Please allow camera access in your browser settings and refresh the page.
          </div>
        )}

        <div className="video-container">
          <video 
            ref={videoRef} 
            className="video-feed" 
            playsInline 
            muted 
            width={CANVAS_WIDTH}
            height={CANVAS_HEIGHT}
          />
          <canvas
            ref={canvasRef}
            className="overlay"
            width={CANVAS_WIDTH}
            height={CANVAS_HEIGHT}
          />
          <div className="center-frame"></div>
        </div>

        <div className="controls">
          <button
            onClick={isWebcamStarted ? stopWebcam : startWebcam}
            disabled={!isOpenCVLoaded}
          >
            {isWebcamStarted ? "Stop Detection" : "Start Detection"}
          </button>
          
          <button
            onClick={() => alert('API would be called here!')}
            disabled={!isCircleStable}
            style={{
              marginLeft: '10px',
              backgroundColor: isCircleStable ? '#28a745' : '#555',
              cursor: isCircleStable ? 'pointer' : 'not-allowed'
            }}
          >
            {isCircleStable ? '✓ Circle Detected - Call API' : 'Waiting for Circle...'}
          </button>
        </div>

        <div className="status">
          {!isWebcamStarted && isOpenCVLoaded && "Click 'Start Detection' to begin"}
          {isWebcamStarted && (
            <>
              <div>
                Live Detection: {predictions.length > 0
                  ? `${predictions.length} circle(s) in frame`
                  : "No circles in frame"}
              </div>
              <div style={{ 
                marginTop: '8px', 
                color: isCircleStable ? '#00ff7f' : '#ff6b6b',
                fontWeight: 'bold'
              }}>
                Status: {isCircleStable ? '✓ Stable - Ready!' : '⧗ Stabilizing...'}
              </div>
            </>
          )}
        </div>

        {/* coin */}
        <img 
        src={coin}
        style={{
          width: '200px',
          height: '200px',
          position: 'fixed',
          bottom: '20px',
        }}
        />
      </div>
    </>
  );
}