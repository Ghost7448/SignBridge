import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22";

const video = document.querySelector("#camera");
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
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm");
  landmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", delegate: "CPU" },
    runningMode: "VIDEO",
    numHands: 1
  });
  loading.hidden = true;
}

async function start() {
  if (active) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    status.textContent = "الكاميرا تحتاج HTTPS أو تشغيلًا محليًا آمنًا.";
    return;
  }
  try {
    startButton.disabled = true;
    status.textContent = "جارٍ تجهيز الكاميرا...";
    await ensureModel();
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    video.srcObject = stream;
    await video.play();
    active = true;
    placeholder.hidden = true;
    stopButton.disabled = false;
    status.textContent = "الكاميرا شغالة — خلي إيدك واضحة";
    requestAnimationFrame(loop);
  } catch (error) {
    console.error(error);
    status.textContent = "تعذر تشغيل الكاميرا أو تحميل نموذج التعرّف. تأكد من السماح بالكاميرا والاتصال بالإنترنت.";
    startButton.disabled = false;
    loading.hidden = true;
  }
}

function loop() {
  if (!active) return;
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const output = landmarker.detectForVideo(video, performance.now());
    const hand = output.landmarks?.[0];
    if (hand) {
      recent.push(classify(hand));
      if (recent.length > 5) recent.shift();
      const counts = recent.reduce((acc, item) => (acc[item] = (acc[item] || 0) + 1, acc), {});
      result.textContent = Object.entries(counts).sort((a,b) => b[1]-a[1])[0][0];
      status.textContent = "تم رصد يد";
    } else {
      recent = [];
      result.textContent = "مش شايف إيد";
      status.textContent = "حط إيدك قدام الكاميرا في إضاءة كويسة";
    }
  }
  requestAnimationFrame(loop);
}

function stop() {
  active = false;
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  recent = [];
  result.textContent = "جاهز";
  status.textContent = "الكاميرا متوقفة";
  placeholder.hidden = false;
  startButton.disabled = false;
  stopButton.disabled = true;
}
startButton.addEventListener("click", start);
stopButton.addEventListener("click", stop);
window.addEventListener("pagehide", stop);
