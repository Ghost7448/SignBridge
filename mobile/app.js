const video = document.querySelector("#camera");
const canvas = document.querySelector("#landmarks");
const ctx = canvas.getContext("2d");
const startButton = document.querySelector("#start");
const stopButton = document.querySelector("#stop");
const result = document.querySelector("#result");
const status = document.querySelector("#status");
const placeholder = document.querySelector("#camera-placeholder");
const loading = document.querySelector("#loading");
let stream = null;
let landmarker = null;
let FilesetResolver = null;
let HandLandmarker = null;
let active = false;
let lastVideoTime = -1;
let recent = [];

function classify(lm) {
  const idx = lm[8].y < lm[6].y - .02;
  const mid = lm[12].y < lm[10].y - .02;
  const ring = lm[16].y < lm[14].y - .02;
  const pinky = lm[20].y < lm[18].y - .02;
  const thumbUp = lm[4].y < lm[3].y - .04 && lm[4].y < lm[2].y - .04;
  const thumbDown = lm[4].y > lm[3].y + .06 && lm[4].y > lm[2].y + .02;
  const folded = [idx, mid, ring, pinky].filter(v => !v).length;
  if (thumbDown && folded >= 3) return "👎 لأ / مش موافق";
  if (thumbUp && folded >= 3) return "👍 تمام";
  if (idx && mid && ring && pinky && thumbUp) return "✋ أهلاً / استنى";
  if (idx && mid && ring && pinky) return "🖖 أربعة";
  if (idx && mid && ring && !pinky) return "🤟 تلاتة";
  if (idx && !mid && !ring && pinky) return "🤘 علامة الروك";
  if (!idx && !mid && !ring && pinky) return "🤙 الخنصر بس";
  if (idx && mid && !ring && !pinky) return "✌️ اتنين";
  if (!idx && !mid && !ring && !pinky) return "✊ قبضة مقفولة";
  if (idx && !mid && !ring && !pinky) return "☝️ واحد";
  return "وضع غير معروف";
}

async function ensureModel() {
  if (landmarker) return;
  loading.hidden = false;
  // Load the AI library only after the camera is already running. A blocked or slow CDN
  // must not prevent the Start button handler from being registered.
  if (!FilesetResolver || !HandLandmarker) {
    ({ FilesetResolver, HandLandmarker } = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/+esm"));
  }
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm"
  );
  landmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "CPU"
    },
    runningMode: "VIDEO",
    numHands: 1
  });
}

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],[0,17]
];

function clearLandmarks() {
  if (!canvas || !ctx) return;
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
}

function drawLandmarks(points) {
  if (!canvas || !ctx || !points || !video.videoWidth || !video.videoHeight) return;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;

  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const width = Math.round(rect.width * dpr);
  const height = Math.round(rect.height * dpr);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  // Match the video's object-fit: contain geometry, including letterbox margins.
  const scale = Math.min(rect.width / video.videoWidth, rect.height / video.videoHeight);
  const shownWidth = video.videoWidth * scale;
  const shownHeight = video.videoHeight * scale;
  const offsetX = (rect.width - shownWidth) / 2;
  const offsetY = (rect.height - shownHeight) / 2;
  const xy = points.map(p => ({
    x: offsetX + p.x * shownWidth,
    y: offsetY + p.y * shownHeight
  }));

  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#45f0b5";
  ctx.lineWidth = 3;
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(xy[a].x, xy[a].y);
    ctx.lineTo(xy[b].x, xy[b].y);
    ctx.stroke();
  }
  for (const p of xy) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#147cff";
    ctx.stroke();
  }
}

async function start() {
  if (active || stream) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    status.textContent = "المتصفح منع الكاميرا. تأكد إن التطبيق محدّث وافتحه من جديد.";
    return;
  }

  startButton.disabled = true;
  status.textContent = "بنفتح الكاميرا...";
  try {
    // Open camera first so users see the live preview even while the AI model downloads.
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: "user" }, width: { ideal: 1280 }, height: { ideal: 720 } }
    });
    video.srcObject = stream;
    await new Promise((resolve, reject) => {
      if (video.readyState >= 2) return resolve();
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(new Error("تعذر عرض صورة الكاميرا"));
      setTimeout(() => reject(new Error("الكاميرا لم تستجب في الوقت المناسب")), 12000);
    });
    await video.play();
    placeholder.hidden = true;
    active = true;
    stopButton.disabled = false;
    status.textContent = "الكاميرا شغالة — بنجهّز التعرّف...";
    requestAnimationFrame(loop);

    try {
      await ensureModel();
      if (active) status.textContent = "الكاميرا شغالة — خلي إيدك واضحة";
    } catch (modelError) {
      console.error("SignBridge model loading failed:", modelError);
      if (active) status.textContent = "الكاميرا شغالة، لكن التعرّف محتاج إنترنت. جرّب اتصال الإنترنت.";
    } finally {
      loading.hidden = true;
      startButton.disabled = active;
    }
  } catch (error) {
    console.error("SignBridge camera error:", error);
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
    video.srcObject = null;
    active = false;
    placeholder.hidden = false;
    loading.hidden = true;
    startButton.disabled = false;
    stopButton.disabled = true;
    if (error?.name === "NotAllowedError" || error?.name === "PermissionDeniedError") {
      status.textContent = "اسمح لـ SignBridge باستخدام الكاميرا من إعدادات الموبايل.";
    } else if (error?.name === "NotFoundError") {
      status.textContent = "مش لاقيين كاميرا متاحة على الجهاز.";
    } else {
      status.textContent = "الكاميرا مشتغلتش. راجع إذن الكاميرا واقفل أي تطبيق بيستخدمها.";
    }
  }
}

function loop() {
  if (!active) return;
  if (landmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    try {
      const output = landmarker.detectForVideo(video, performance.now());
      const hand = output.landmarks?.[0];
      if (hand) {
        drawLandmarks(hand);
        recent.push(classify(hand));
        if (recent.length > 5) recent.shift();
        const counts = recent.reduce((acc, item) => (acc[item] = (acc[item] || 0) + 1, acc), {});
        result.textContent = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
        status.textContent = "تم رصد يد";
      } else {
        clearLandmarks();
        recent = [];
        result.textContent = "مش شايف إيد";
        if (!loading.hidden) status.textContent = "بنحمّل التعرّف...";
        else status.textContent = "حط إيدك قدام الكاميرا في إضاءة كويسة";
      }
    } catch (error) {
      console.error("SignBridge detection error:", error);
    }
  }
  requestAnimationFrame(loop);
}

function stop() {
  active = false;
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  clearLandmarks();
  recent = [];
  result.textContent = "جاهز";
  status.textContent = "الكاميرا متوقفة";
  placeholder.hidden = false;
  loading.hidden = true;
  startButton.disabled = false;
  stopButton.disabled = true;
}
startButton.addEventListener("click", start);
stopButton.addEventListener("click", stop);
window.addEventListener("pagehide", stop);
