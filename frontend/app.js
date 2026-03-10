const socket = new WebSocket("ws://localhost:5000");

let recorder;
let stream;

const startBtn = document.getElementById("start");
const stopBtn = document.getElementById("stop");
const transcriptBox = document.getElementById("transcript");

startBtn.onclick = async () => {

    console.log("Start button clicked");

    try {

        stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        console.log("Microphone access granted");

    } catch (err) {

        console.log("Microphone permission denied");
        return;

    }

    socket.send(JSON.stringify({ type: "start-recording" }));

    recorder = new MediaRecorder(stream);

    recorder.ondataavailable = (event) => {

        if (event.data.size > 0) {

            socket.send(event.data);

        }

    };

    recorder.start(1000); // send audio every 1 second

};

stopBtn.onclick = () => {

    if (recorder) recorder.stop();

    socket.send(JSON.stringify({ type: "stop-recording" }));

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }

};

socket.onmessage = (event) => {

    const data = JSON.parse(event.data);

    if (data.type === "transcription") {

        transcriptBox.innerText = data.text;

    }

};