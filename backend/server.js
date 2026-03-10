const http = require("http");
const WebSocket = require("ws");
const fs = require("fs");
const { exec } = require("child_process");

const server = http.createServer();
const wss = new WebSocket.Server({ server });

let currentFile = null;
let lastRunTime = 0;

wss.on("connection", (ws) => {

    ws.on("message", (message) => {

        try {

            const data = JSON.parse(message);

            if (data.type === "start-recording") {

                currentFile = `recording_${Date.now()}.webm`;

                fs.writeFileSync(currentFile, "");

                lastRunTime = Date.now();

            }

            if (data.type === "stop-recording") {

                currentFile = null;

            }

        } catch {

            if (!currentFile) return;

            fs.appendFileSync(currentFile, message);

            const now = Date.now();

            // run whisper every 3 seconds
            if (now - lastRunTime > 3000) {

                lastRunTime = now;

                exec(`python transcribe.py ${currentFile}`, (error, stdout) => {

                    if (error) return;

                    if (stdout) {

                        ws.send(JSON.stringify({
                            type: "transcription",
                            text: stdout.trim()
                        }));

                    }

                });

            }

        }

    });

});

server.listen(5000, () => {
    console.log("Server running on port 5000");
});