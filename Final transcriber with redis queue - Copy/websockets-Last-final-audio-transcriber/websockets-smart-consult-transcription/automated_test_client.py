import asyncio
import argparse
import json
import uuid
import wave
import time
import sys
import os

try:
    import websockets
except ImportError:
    print("Please install websockets client: pip install websockets")
    sys.exit(1)

async def stream_audio_file(ws_url: str, audio_file_path: str):
    session_id = str(uuid.uuid4())
    ws_uri = f"{ws_url}/ws/consultation/{session_id}"
    
    print(f"\n[{time.strftime('%H:%M:%S')}] [Session {session_id}] Connecting to {ws_uri}...")
    
    try:
        wf = wave.open(audio_file_path, 'rb')
        framerate = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        
        if framerate != 16000 or channels != 1 or sampwidth != 2:
            print(f"WARNING: Audio should ideally be 16kHz, mono, 16-bit PCM. Found: {framerate}Hz, {channels}ch, {sampwidth}byte")
    except Exception as e:
        print(f"Error opening {audio_file_path}. Ensure it is a valid .wav file: {e}")
        return

    try:
        async with websockets.connect(ws_uri) as websocket:
            print(f"[{time.strftime('%H:%M:%S')}] [Session {session_id}] Connected. Starting real-time mock stream...")
            
            # Send standard handshake
            await websocket.send(json.dumps({"type": "start", "sample_rate_hz": framerate}))
            
            # Send chunks mimicking 100ms of audio at a time
            chunk_duration_sec = 0.1
            bytes_per_sec = framerate * channels * sampwidth
            chunk_size = int(chunk_duration_sec * bytes_per_sec)
            
            async def receive_transcripts():
                try:
                    while True:
                        msg = await websocket.recv()
                        if isinstance(msg, str):
                            data = json.loads(msg)
                            msg_type = data.get("type")
                            text = data.get("text", "")
                            
                            if msg_type == "final":
                                print(f"\n[FINAL] {text}")
                            elif msg_type == "partial":
                                # Overwrite the current line for partials
                                sys.stdout.write(f"\r[PARTIAL] {text[:80]}...{' '*20}")
                                sys.stdout.flush()
                            elif msg_type == "warning":
                                print(f"\n[WARNING] {text}")
                            elif msg_type == "end":
                                print(f"\n[{time.strftime('%H:%M:%S')}] [Session {session_id}] Server gracefully disconnected. Transcript saved: {data.get('transcript_file')}")
                                break
                except websockets.exceptions.ConnectionClosed:
                    print(f"\n[{time.strftime('%H:%M:%S')}] [Session {session_id}] Connection closed by server.")

            recv_task = asyncio.create_task(receive_transcripts())
            
            start_time = time.time()
            audio_streamed = 0.0
            
            # Send audio continuously 
            while True:
                data = wf.readframes(chunk_size // sampwidth)
                if not data:
                    break
                    
                await websocket.send(data)
                audio_streamed += chunk_duration_sec
                
                # Sleep to pace the data exactly to real-time speed.
                # This guarantees we test the backend robustness accurately without flooding it.
                elapsed = time.time() - start_time
                if audio_streamed > elapsed:
                    await asyncio.sleep(audio_streamed - elapsed)
                    
            print(f"\n[{time.strftime('%H:%M:%S')}] [Session {session_id}] Finished streaming source file. Waiting for final server flush...")
            await websocket.send("END")
            
            # Wait securely for the server to process the final queue
            await recv_task
            
    except Exception as e:
        print(f"\n[{time.strftime('%H:%M:%S')}] [Session {session_id}] WebSocket Error: {e}")


async def run_continuous_test(ws_url: str, audio_file_path: str, loop_count: int):
    print(f"Starting Automated Test Suite.")
    print(f"Target WS URL: {ws_url}")
    print(f"Audio File: {audio_file_path}")
    print(f"Cycles: {loop_count if loop_count > 0 else 'Infinite'}\n")
    
    cycle = 1
    while True:
        print(f"=== Starting Test Cycle {cycle} ===")
        await stream_audio_file(ws_url, audio_file_path)
        print(f"=== Cycle {cycle} Complete ===\n")
        
        if 0 < loop_count <= cycle:
            print("Finished all requested cycles. Exiting.")
            break
            
        print("Waiting 4 seconds before next cycle (simulating next patient entry)...")
        await asyncio.sleep(4)
        cycle += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuous Automated Whisper WebSocket Tester")
    parser.add_argument("audio_file", help="Path to the .wav file to stream sequentially")
    parser.add_argument("--url", default="ws://localhost:8000", help="WebSocket server URL")
    parser.add_argument("--loops", type=int, default=0, help="Number of times to replay the file. 0 for infinite.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found at '{args.audio_file}'")
        sys.exit(1)
        
    try:
        asyncio.run(run_continuous_test(args.url, args.audio_file, args.loops))
    except KeyboardInterrupt:
        print("\nTest manually interrupted by user.")
