import * as handpose from '@tensorflow-models/handpose';
import * as tf from '@tensorflow/tfjs';
import React, { useEffect, useRef, useState } from 'react';

export default function NailKYCCamera() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const capturedCanvasRef = useRef(null);
  
  const [model, setModel] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [detectionStatus, setDetectionStatus] = useState('Initializing...');
  const [fingersDetected, setFingersDetected] = useState(0);
  const [capturedImage, setCapturedImage] = useState(null);
  const [nailMeasurements, setNailMeasurements] = useState(null);
  const [countdown, setCountdown] = useState(null);

  // Initialize camera and load model
  useEffect(() => {
    let animationFrame;
    let countdownTimer;

    const setupCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { 
            facingMode: 'user',
            width: { ideal: 1280 },
            height: { ideal: 720 }
          }
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current.play();
          };
        }
      } catch (err) {
        setDetectionStatus('Camera access denied. Please allow camera.');
        console.error('Camera error:', err);
      }
    };

    const loadModel = async () => {
      try {
        setDetectionStatus('Loading AI model...');
        await tf.ready();
        const handposeModel = await handpose.load();
        setModel(handposeModel);
        setIsLoading(false);
        setDetectionStatus('Ready! Show your hand with all 5 fingers spread.');
      } catch (err) {
        setDetectionStatus('Error loading model. Please refresh.');
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

  // Detect hand and fingers
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
        
        // Set canvas size to match video
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (predictions.length > 0) {
          const hand = predictions[0];
          const landmarks = hand.landmarks;
          
          // Draw hand skeleton
          drawHand(ctx, landmarks);
          
          // Check if all 5 fingers are visible
          const fingers = detectFingers(landmarks);
          setFingersDetected(fingers.length);
          
          if (fingers.length === 5 && !isCapturing && !capturedImage) {
            setDetectionStatus('Perfect! All 5 fingers detected. Capturing in...');
            isCapturing = true;
            
            // Countdown before capture
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
          } else if (fingers.length < 5 && !capturedImage) {
            setDetectionStatus(`${fingers.length}/5 fingers detected. Spread all fingers!`);
            if (countdownTimer) {
              clearTimeout(countdownTimer);
              setCountdown(null);
              isCapturing = false;
            }
          }
        } else {
          setFingersDetected(0);
          if (!capturedImage) {
            setDetectionStatus('No hand detected. Show your hand to camera.');
          }
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

  const drawHand = (ctx, landmarks) => {
    // Draw finger connections
    const fingerJoints = [
      [0, 1, 2, 3, 4],     // Thumb
      [0, 5, 6, 7, 8],     // Index
      [0, 9, 10, 11, 12],  // Middle
      [0, 13, 14, 15, 16], // Ring
      [0, 17, 18, 19, 20]  // Pinky
    ];

    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;

    fingerJoints.forEach(finger => {
      for (let i = 0; i < finger.length - 1; i++) {
        const [x1, y1] = landmarks[finger[i]];
        const [x2, y2] = landmarks[finger[i + 1]];
        
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    });

    // Draw landmarks (joints)
    landmarks.forEach(([x, y], index) => {
      ctx.fillStyle = index % 4 === 0 ? '#ff0000' : '#00ff00';
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fill();
    });

    // Draw nail regions (fingertips)
    const fingertips = [4, 8, 12, 16, 20];
    ctx.strokeStyle = '#ffff00';
    ctx.lineWidth = 3;
    
    fingertips.forEach(tip => {
      const [x, y] = landmarks[tip];
      ctx.beginPath();
      ctx.arc(x, y, 15, 0, 2 * Math.PI);
      ctx.stroke();
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
      
      // Calculate if finger is extended
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
    
    // Draw video frame
    ctx.drawImage(video, 0, 0);
    
    // Draw overlay
    drawHand(ctx, landmarks);
    
    const imageData = capturedCanvas.toDataURL('image/png');
    setCapturedImage(imageData);
    
    // Calculate nail measurements
    const measurements = calculateNailMeasurements(landmarks);
    setNailMeasurements(measurements);
    
    setDetectionStatus('Image captured successfully!');
  };

  const calculateNailMeasurements = (landmarks) => {
    const fingertips = [4, 8, 12, 16, 20];
    const fingerNames = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'];
    const measurements = {};
    
    fingertips.forEach((tip, index) => {
      const tipBehind = tip - 1;
      const [tipX, tipY] = landmarks[tip];
      const [behindX, behindY] = landmarks[tipBehind];
      
      // Approximate nail dimensions
      const length = Math.sqrt(
        Math.pow(tipX - behindX, 2) + Math.pow(tipY - behindY, 2)
      );
      
      measurements[fingerNames[index]] = {
        length: length.toFixed(2),
        position: { x: tipX.toFixed(0), y: tipY.toFixed(0) }
      };
    });
    
    return measurements;
  };

  const resetCapture = () => {
    setCapturedImage(null);
    setNailMeasurements(null);
    setDetectionStatus('Ready! Show your hand with all 5 fingers spread.');
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 p-4">
      <div className="max-w-4xl w-full">
        <h1 className="text-4xl font-bold text-white mb-2 text-center">
          Nail KYC Scanner
        </h1>
        <p className="text-gray-400 text-center mb-6">
          AI-Powered Hand & Nail Detection
        </p>

        {/* Status Bar */}
        <div className="bg-gray-800 rounded-lg p-4 mb-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={`w-3 h-3 rounded-full ${
                fingersDetected === 5 ? 'bg-green-500' : 'bg-yellow-500'
              } animate-pulse`} />
              <span className="text-white font-medium">{detectionStatus}</span>
            </div>
            {countdown && (
              <div className="text-4xl font-bold text-green-400 animate-pulse">
                {countdown}
              </div>
            )}
            <div className="text-white">
              Fingers: <span className={`font-bold ${
                fingersDetected === 5 ? 'text-green-400' : 'text-yellow-400'
              }`}>{fingersDetected}/5</span>
            </div>
          </div>
        </div>

        {/* Camera View */}
        {!capturedImage ? (
          <div className="relative bg-black rounded-lg overflow-hidden shadow-2xl">
            <video
              ref={videoRef}
              className="w-full h-auto"
              autoPlay
              playsInline
              muted
            />
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 w-full h-full"
            />
            {isLoading && (
              <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-75">
                <div className="text-white text-xl">Loading AI Model...</div>
              </div>
            )}
          </div>
        ) : (
          <div className="bg-black rounded-lg overflow-hidden shadow-2xl">
            <img src={capturedImage} alt="Captured" className="w-full h-auto" />
          </div>
        )}

        {/* Hidden canvas for capture */}
        <canvas ref={capturedCanvasRef} className="hidden" />

        {/* Nail Measurements */}
        {nailMeasurements && (
          <div className="mt-6 bg-gray-800 rounded-lg p-6">
            <h2 className="text-2xl font-bold text-white mb-4">
              Nail Measurements
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(nailMeasurements).map(([finger, data]) => (
                <div key={finger} className="bg-gray-700 rounded p-4">
                  <h3 className="text-lg font-semibold text-green-400 mb-2">
                    {finger}
                  </h3>
                  <p className="text-white">
                    Length: <span className="text-yellow-400">{data.length}px</span>
                  </p>
                  <p className="text-white text-sm mt-1">
                    Position: ({data.position.x}, {data.position.y})
                  </p>
                </div>
              ))}
            </div>
            <button
              onClick={resetCapture}
              className="mt-6 w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg transition"
            >
              Capture Again
            </button>
          </div>
        )}

        {/* Instructions */}
        <div className="mt-6 bg-gray-800 rounded-lg p-4">
          <h3 className="text-white font-semibold mb-2">Instructions:</h3>
          <ul className="text-gray-300 text-sm space-y-1">
            <li>• Spread all 5 fingers clearly in front of camera</li>
            <li>• Keep your hand steady and well-lit</li>
            <li>• The system will auto-capture when all fingers detected</li>
            <li>• Green circles indicate fingertips (nail areas)</li>
          </ul>
        </div>
      </div>
    </div>
  );
}