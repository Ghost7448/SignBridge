import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

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
let active = false;
let lastVideoTime = -1;
let recent = [];

const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],
  [10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],
  [18,19],[19,20],[0,17]
];

function classify(lm) {
  const i = lm[8].y < lm[6].y - .02;
  const m = lm[12].y < lm[10].y - .02;
  const r = lm[16].y < lm[14].y - .02;
  const p = lm[20].y < lm[18].y - .02;
  const tu = lm[4].y < lm[3].y - .04 && lm[4].y < lm[2].y - .04;
  const td = lm[4].y > lm[3].y + .06 && lm[4].y > lm[2].y + .02;
  const fold = [i,m,r,p].filter(x => !x).length;
  if (td && fold >= 3) return "👎 لأ / مش موافق";
  if (tu && fold >= 3) return "👍 تمام";
  if (i && m && r && p && tu) return "✋ أهلاً / استنى";
  if (i && m && r && p) return "🖖 أربعة";
  if (i && m && r && !p) return "🤟 تلاتة";
  if (i && !m && !r && p) return "🤘 علامة الروك";
  if (!i && !m && !r && p) return "🤙 الخنصر بس";
  if (i && m && !r && !p) return "✌️ اتنين";
  if (!i && !m && !r && !p) return "✊ قبضة مقفولة";
  if (i && !m && !r && !p) return "☝️ واحد";
  return "وضع غير معروف";
}

async function ensureModel() {
  if (landmarker) return;
  loading.hidden = false;
  status.textContent = "بنجهّز التعرّف على اليد...";
  const vision = await FilesetResolver.forVisionTasks("./wasm");
  landmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "./models/hand_landmarker.task", delegate: "CPU" },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: .5,
    minHandPresenceConfidence: .5,
    minTrackingConfidence: .5
  });
}

/*
 * Match the overlay to the actual video image rectangle.
 * The video element fills a 4:3 container with object-fit:contain, so
 * normalized landmarks must be offset for any letterboxing before drawing.
 */
function canvasMetrics() {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const cssWidth = rect.width;
  const cssHeight = rect.height;
  if (!cssWidth || !cssHeight || !video.videoWidth || !video.videoHeight) return null;

  const backingWidth = Math.round(cssWidth * dpr);
  const backingHeight = Math.round(cssHeight * dpr);
  if (canvas.width !== backingWidth || canvas.height !== backingHeight) {
    canvas.width = backingWidth;
    canvas.height = backingHeight;
  }

  const scale = Math.min(cssWidth / video.videoWidth, cssHeight / video.videoHeight);
  const imageWidth = video.videoWidth * scale;
  const imageHeight = video.videoHeight * scale;
  return {
    dpr,
    width: backingWidth,
    height: backingHeight,
    scale: scale * dpr,
    offsetX: ((cssWidth - imageWidth) / 2) * dpr,
    offsetY: ((cssHeight - imageHeight) / 2) * dpr
  };
}

function clearPoints() {
  const m = canvasMetrics();
  if (m) ctx.clearRect(0, 0, m.width, m.height);
}

function drawPoints(points) {
  if (!points || points.length !== 21 || !video.videoWidth || !video.videoHeight) return;
  const m = canvasMetrics();
  if (!m) return;

  ctx.clearRect(0, 0, m.width, m.height);
  const xy = points.map(p => ({
    x: m.offsetX + p.x * video.videoWidth * m.scale,
    y: m.offsetY + p.y * video.videoHeight * m.scale
  }));

  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#45f0b5";
  ctx.lineWidth = Math.max(2.5 * m.dpr, m.width / 320);
  for (const [a, b] of CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(xy[a].x, xy[a].y);
    ctx.lineTo(xy[b].x, xy[b].y);
    ctx.stroke();
  }
  for (const p of xy) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, Math.max(3.5 * m.dpr, m.width / 420), 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.fill();
    ctx.lineWidth = 1.8 * m.dpr;
    ctx.strokeStyle = "#147cff";
    ctx.stroke();
  }
}

async function start() {
  if (active || stream) return;
  startButton.disabled = true;
  status.textContent = "بنفتح الكاميرا...";
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("الكاميرا غير مدعومة");
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: "user" },
        width: { ideal: 640 },
        height: { ideal: 480 },
        aspectRatio: { ideal: 4 / 3 }
      }
    });
    video.srcObject = stream;
    await new Promise((resolve, reject) => {
      if (video.readyState >= 2 && video.videoWidth) return resolve();
      const timer = setTimeout(() => reject(new Error("الكاميرا ما بعتتش صورة")), 12000);
      video.onloadedmetadata = () => { clearTimeout(timer); resolve(); };
      video.onerror = () => { clearTimeout(timer); reject(new Error("تعذر عرض صورة الكاميرا")); };
    });
    await video.play();
    canvasMetrics();
    placeholder.hidden = true;
    active = true;
    stopButton.disabled = false;
    status.textContent = "الكاميرا شغالة — بنجهّز التعرّف";
    requestAnimationFrame(loop);
    await ensureModel();
    if (active) status.textContent = "قرّب إيدك وخليها كاملة جوه الكادر";
  } catch (e) {
    console.error("SignBridge start error", e);
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = null;
    video.srcObject = null;
    active = false;
    placeholder.hidden = false;
    status.textContent = e?.name === "NotAllowedError"
      ? "اسمح للتطبيق باستخدام الكاميرا من إعدادات الموبايل"
      : "التعرّف مش اشتغل: " + (e?.message || "راجع الكاميرا");
  } finally {
    loading.hidden = true;
    startButton.disabled = active;
    stopButton.disabled = !active;
  }
}

function loop() {
  if (!active) return;
  if (landmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    try {
      const out = landmarker.detectForVideo(video, performance.now());
      const hand = out.landmarks?.[0];
      if (hand) {
        drawPoints(hand);
        recent.push(classify(hand));
        if (recent.length > 5) recent.shift();
        const counts = recent.reduce((a, x) => (a[x] = (a[x] || 0) + 1, a), {});
        result.textContent = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
        status.textContent = "تم رصد اليد";
      } else {
        clearPoints();
        recent = [];
        result.textContent = "مش شايف إيد";
        status.textContent = "حط إيدك قدام الكاميرا في إضاءة كويسة";
      }
    } catch (e) {
      console.error("SignBridge detection error", e);
      status.textContent = "حصلت مشكلة في اكتشاف اليد";
    }
  }
  requestAnimationFrame(loop);
}

function stop() {
  active = false;
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null;
  video.srcObject = null;
  clearPoints();
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
window.addEventListener("resize", clearPoints);
