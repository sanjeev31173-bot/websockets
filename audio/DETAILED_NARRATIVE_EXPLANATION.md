# Complete Real-Time Transcription Flow - Detailed Narrative Explanation

## The Complete Journey from Button Click to Final Transcript

### The Initial State: System at Rest

Before any user interaction, the system exists in a state of quiet readiness. The FastAPI server is running on port 8000, its WebSocket endpoint `/ws/consultation/{session_id}` is listening for connections, but no active sessions exist. In the browser, the frontend page is loaded with all its JavaScript modules initialized, but no audio processing is occurring. The WebSocket variable is null, the recording flag is false, and all audio-related objects (AudioContext, MediaStream, ScriptProcessor) are uninitialized. The UI displays a green "Start recording" button, a red disconnected status indicator, and an empty transcript box ready to receive text. On the backend, the Whisper model remains unloaded, conserving memory until needed, and the transcripts directory sits empty, waiting for completed consultation records.

### The Spark: User Clicks "Start Recording"

The moment the user clicks the "Start recording" button, a cascade of events begins. The browser's JavaScript event loop captures the click event at approximately T=0ms and immediately executes the click handler. The handler first checks the `isRecording` flag, which is currently false, allowing the recording process to proceed. Within milliseconds, the system generates a new universally unique identifier using `crypto.randomUUID()`, creating a 128-bit random number like "550e8400-e29b-41d4-a716-446655440000" that will serve as the unique session identifier for this entire consultation. This ID is immediately displayed in the UI, giving the user a reference for their session, and it becomes the cornerstone of the WebSocket URL that will be constructed moments later.

### The Gateway: Requesting Microphone Access

At approximately T=5ms, the system makes its first critical request to the user: `navigator.mediaDevices.getUserMedia({audio: true, video: false})`. This triggers the browser's security mechanism, causing a permission dialog to appear asking the user to grant microphone access. This is a pivotal moment - if the user denies permission, the entire process fails gracefully with an error message asking the user to check microphone permissions. However, in our successful scenario, the user clicks "Allow", and the browser responds by creating a MediaStream object that represents the audio track from the user's default microphone. This MediaStream contains audio data at the device's native sample rate (typically 44.1kHz or 48kHz) and becomes the source of all subsequent audio processing. The browser also activates the AudioContext, which was previously in a suspended state due to autoplay restrictions, because user interaction (the click) has now satisfied the security requirements for audio playback.

### The Audio Engine: Building the Processing Pipeline

Once microphone access is granted (typically around T=5000ms after the initial click, accounting for user interaction time), the system rapidly constructs its audio processing infrastructure. At T=5010ms, it creates an AudioContext with a forced 16kHz sample rate, which is crucial because this matches the Whisper model's expected input format and eliminates the need for complex resampling later. The AudioContext serves as the central hub for all audio operations, managing the audio processing graph that will transform raw microphone input into transcribed text. Immediately following this, at T=5020ms, the system creates two critical nodes: a MediaStreamSource that wraps the granted MediaStream, connecting the physical microphone to the Web Audio API, and a ScriptProcessor with a 4096-sample buffer size that will handle the real-time audio processing. The 4096-sample buffer is carefully chosen because it represents exactly 256 milliseconds of audio at 16kHz (4096 ÷ 16000 = 0.256 seconds), providing a balance between low latency and processing efficiency.

### The Connection: Establishing WebSocket Communication

Simultaneously with the audio pipeline setup, the system initiates network communication. At T=5025ms, it constructs the WebSocket URL by combining the protocol, host, port, and the unique session ID: "ws://127.0.0.1:8000/ws/consultation/550e8400-e29b-41d4-a716-446655440000". The browser then creates a WebSocket object and initiates a TCP connection to the server. This triggers a complex handshake process where the browser sends an HTTP Upgrade request with special WebSocket headers, and the FastAPI server responds with a 101 Switching Protocols response, converting the HTTP connection into a persistent WebSocket connection. By T=5030ms, the WebSocket connection is fully established and transitions to the OPEN state, ready for bidirectional communication.

### The Backend Session: Creating State Management

On the server side, when the WebSocket connection request arrives at T=5027ms, FastAPI routes it to the consultation_ws endpoint handler. The server extracts the session ID from the URL and immediately creates an AudioSession object, which becomes the central state manager for this entire consultation. This AudioSession initializes several critical components: a bounded audio queue with a maximum capacity of 400 chunks (preventing memory exhaustion), a transcript queue capable of holding 2000 messages, and an empty list that will accumulate the complete transcript. The session also records its creation timestamp and sets the default sample rate to 16kHz. At T=5031ms, the server launches the transcription worker as a background asyncio task, which will independently process audio data as it arrives. This worker immediately attempts to load the Whisper model if it hasn't been loaded already - a process that takes 5-10 seconds on the first run but is nearly instantaneous on subsequent sessions.

### The Task Orchestra: Coordinating Concurrent Operations

The server then creates two additional concurrent tasks that form the core of the real-time processing pipeline. The producer task, created at T=5032ms, continuously listens for incoming WebSocket messages and handles both binary audio frames and text messages like the "END" command. The consumer task, also created at T=5032ms, waits for transcript updates in the transcript queue and sends them back to the client via WebSocket. These two tasks run concurrently with the transcription worker, creating a three-part processing pipeline: producer receives audio, transcription worker processes it, and consumer sends results. The `asyncio.gather()` call ensures that these tasks run together and that the session is properly cleaned up when any task completes or fails.

### The Audio Flow: Real-Time Processing Begins

At T=5035ms, the WebSocket connection's `onopen` event fires on the frontend, triggering a cascade of UI updates. The connection indicator changes from red to green, the button text updates to "Stop recording", and the status message changes to "Recording… speak as the patient." The system also sends a configuration message to the backend containing the AudioContext's sample rate (16000Hz), which the backend stores in the session object for potential resampling operations. Then, at T=5039ms, the system sets up the audio processing callback that will execute every 256ms, and at T=5040ms, it connects the audio processing graph by linking the MediaStreamSource to the ScriptProcessor and the ScriptProcessor to the AudioContext destination.

### The First Sound: Audio Processing in Action

At T=5041ms, exactly 256ms after the audio graph connection, the first `onaudioprocess` event fires. This is where the magic begins. The ScriptProcessor provides an AudioProcessingEvent containing an input buffer with 4096 float32 samples representing the first 256 milliseconds of microphone input. Let's imagine the user says "Hello, doctor" - the first buffer might contain the "He" part of "Hello". The system extracts this audio data as a Float32Array with values ranging from -1.0 to 1.0, representing the sound pressure waves captured by the microphone. Since the AudioContext is already at 16kHz, no resampling is needed, and the system proceeds directly to PCM16 conversion.

### The Digital Transformation: From Float to Integer

The conversion process transforms each floating-point sample into a 16-bit integer. For each sample value in the Float32Array, the system first clamps it to ensure it stays within the -1.0 to 1.0 range, then scales it by multiplying by 32767 for positive values or 32768 for negative values. This scaling converts the floating-point representation to the integer format that Whisper expects. For example, a sample value of 0.5 becomes 16383, while -0.75 becomes -24576. The system uses a DataView to write these integers in little-endian format into an ArrayBuffer, creating an 8192-byte binary blob (4096 samples × 2 bytes per sample). This binary data represents the exact digital audio that will be sent to the server.

### The Network Journey: WebSocket Transmission

At T+3ms within the audio processing callback (approximately T+5044ms overall), the system sends this 8192-byte ArrayBuffer via WebSocket. The WebSocket frames this data as a binary message and sends it over the TCP connection to the server. On localhost, this transmission takes approximately 1ms, but in a real-world scenario, it would depend on network conditions. The message arrives at the backend's producer task, which is waiting in the `websocket.receive()` call. The producer receives the binary frame and immediately calls `session.add_audio_chunk()` to queue the audio data.

### The Queue Management: Protecting System Resources

The audio chunk enters the session's bounded audio queue, which can hold up to 400 chunks. The queue operation uses `asyncio.wait_for()` with a 2-second timeout to protect against memory exhaustion. If the queue is full (which would happen if the transcription worker is processing slower than audio is arriving), the system waits up to 2 seconds for space to become available. If no space becomes available, it gracefully closes the session to prevent unbounded memory growth. In our normal scenario, the queue has plenty of space, and the audio chunk is successfully queued within microseconds.

### The Transcription Engine: Processing the Audio Stream

Meanwhile, the transcription worker task is running in its own async loop, waiting for audio data through the `session.audio_iter()` async iterator. As soon as the first audio chunk becomes available in the queue, the transcription worker retrieves it and begins processing. The worker converts the raw bytes into a NumPy int16 array using `np.frombuffer()`, then adds this array to a rolling buffer implemented as a Python deque. This rolling buffer maintains exactly 30 seconds of audio data - as new chunks are added, the oldest chunks are removed to keep the buffer size constant. This design ensures that the system can handle arbitrarily long consultations without consuming unlimited memory.

### The Timing Logic: When to Transcribe

The transcription worker doesn't transcribe every audio chunk immediately. Instead, it uses sophisticated timing logic to balance responsiveness with efficiency. It waits until it has at least 6 seconds of audio data (96,000 samples at 16kHz) before attempting the first transcription. This minimum window ensures that Whisper has enough context to produce accurate results. After the first transcription, it transcribes every 2.5 seconds, providing regular updates without overwhelming the CPU. The worker uses `time.monotonic()` for accurate timing and maintains a `last_transcribe_at` timestamp to enforce the interval.

### The Whisper Inference: Converting Audio to Text

When the timing conditions are met (first at T+11000ms, when 6 seconds of audio have accumulated), the worker prepares the audio for Whisper. It concatenates all chunks in the rolling buffer into a single NumPy array, converts the int16 values to float32 by dividing by 32768.0, and calls the Whisper model's `transcribe()` method. This is the most computationally intensive step - the base Whisper model processes the 30-second audio buffer, performing complex neural network computations to convert the audio signal into text. On a CPU, this typically takes about 500ms, during which the model analyzes the audio patterns, identifies phonemes, words, and sentence structures, and produces a structured result containing segments with timestamps.

### The Segment Processing: Extracting Meaningful Text

Whisper returns a dictionary containing multiple segments, each with start and end timestamps and the transcribed text for that segment. The transcription worker processes these segments sequentially, using sophisticated duplicate prevention logic. It calculates the absolute end time for each segment by adding the segment's end time to the rolling buffer's start time. If this absolute end time is less than or equal to the last emitted end time plus a 50ms tolerance, the segment is skipped as a duplicate. This prevents the same audio from being transcribed multiple times as the rolling buffer slides forward. For new segments, the worker extracts the text, strips whitespace, and adds it to the list of emitted texts. It also updates the `last_emitted_end_sec` to track the furthest point in the audio that has been transcribed.

### The Incremental Processing: Preventing Re-Transcription

After extracting the text segments, the worker performs crucial cleanup to prevent re-transcription of old audio. It calculates how many samples were consumed by the transcription (based on the last segment's end time) and removes that many samples from the beginning of the rolling buffer. This is done carefully - if a chunk is only partially consumed, it's split and the remaining portion is kept. This incremental processing ensures that as the consultation progresses, the system always focuses on transcribing new audio rather than repeatedly processing the same content.

### The Transcript Emission: Sending Results Back

For each unique text segment extracted, the worker calls `session.push_partial_transcript()`. This method adds the text to the session's complete transcript list (which accumulates everything for the final file) and attempts to add it to the transcript queue for immediate transmission to the client. The transcript queue is also bounded (2000 messages), and if it's full, the system drops new messages rather than crashing - this graceful degradation ensures that the system continues processing even if the network is slow.

### The Real-Time Display: User Sees Results

Meanwhile, the consumer task on the backend is waiting for messages in the transcript queue. As soon as a transcript message appears, the consumer retrieves it and sends it to the client via WebSocket using `websocket.send_text()`. This text message travels back over the WebSocket connection to the browser, where the `onmessage` event handler receives it. The handler attempts to parse the message as JSON - if it's the final "end" message, it updates the status to "Transcription finished." Otherwise, it treats the message as transcript text and appends it to the transcript display area in the UI, adding a space after each segment for readability. The system also automatically scrolls the transcript area to the bottom, ensuring the user always sees the most recent text.

### The Continuous Loop: Building the Complete Picture

This entire process - audio capture, WebSocket transmission, queue management, transcription, and display - repeats continuously every 256ms for audio chunks and every 2.5 seconds for transcription updates. As the user continues speaking, each 256ms audio chunk is captured and sent to the server, where it accumulates in the rolling buffer. Every 2.5 seconds, the system transcribes the latest 30-second window, extracts any new text segments that haven't been seen before, and sends them to the client. The user sees the transcript building up in real-time, with new text appearing every few seconds as they speak.

### The User's Example: "Hello, doctor, I've been experiencing headaches"

Let's trace through a concrete example. Suppose the user says "Hello, doctor, I've been experiencing headaches for the past week." The first audio chunk (T+5041ms) might contain just "Hel", the second chunk (T+5297ms) contains "lo, do", and so on. These chunks accumulate in the rolling buffer. At T+11000ms, when 6 seconds of audio have been collected, the system transcribes the first portion, perhaps producing "Hello, doctor I've been". This text appears on the user's screen. Two and a half seconds later (T+13500ms), the system transcribes again, this time with more context, and might produce "experiencing headaches for the". The incremental processing ensures that only the new portion "experiencing headaches for the" is sent to the client, avoiding duplication. This continues until the complete sentence appears as "Hello, doctor, I've been experiencing headaches for the past week."

### The Termination: Ending the Session Gracefully

When the user clicks "Stop recording", the system initiates a graceful shutdown sequence. The frontend first disconnects the ScriptProcessor from the audio graph, stopping further audio processing callbacks. It then closes the AudioContext, which releases all audio resources and stops the microphone. The MediaStream's tracks are stopped, which physically turns off the microphone. Finally, the frontend sends an "END" message to the backend via WebSocket, signaling that no more audio will be coming.

### The Final Processing: Completing the Transcription

When the backend receives the "END" message, the producer task marks the session as closed, which sends an empty byte sentinel to the audio queue. The transcription worker receives this sentinel, processes any remaining audio in the rolling buffer one final time, and then calls `session.mark_transcription_finished()`. This places an "__END__" sentinel in the transcript queue, signaling the consumer task that transcription is complete.

### The Persistence: Saving the Results

The consumer task, upon receiving the "__END__" sentinel, performs the final persistence step. It takes the complete transcript list, filters out any warning messages, joins all the text segments with spaces, and saves the result to a file in the transcripts directory. The filename uses the session ID, creating a permanent record of the consultation. For our example, this might create a file named "550e8400-e29b-41d4-a716-446655440000.txt" containing the text "Hello, doctor, I've been experiencing headaches for the past week."

### The Completion: Final Results on Screen

The consumer task then sends a final JSON message to the client: `{"type":"end","transcript_file":"transcripts/550e8400-e29b-41d4-a716-446655440000.txt"}`. When the frontend receives this message, it updates the status to "Transcription finished" and resets the UI for the next recording session. The WebSocket connection closes, the connection indicator turns red again, and the button text returns to "Start recording." The user can now see their complete transcript on screen and know that a permanent file has been saved.

### The Resource Cleanup: System Returns to Rest

Throughout this entire process, the system carefully manages resources to prevent memory leaks and ensure stability. The bounded queues prevent unlimited memory consumption, the rolling buffer maintains a constant 30-second window regardless of consultation length, and all audio resources are properly released when the session ends. The Whisper model remains loaded in memory for the next session, avoiding the 5-10 second loading time on subsequent consultations. The system returns to its initial state of quiet readiness, waiting for the next user to click "Start recording" and begin the entire journey again.

This complete flow demonstrates how the system transforms live speech into a permanent text record in real-time, balancing performance, accuracy, and resource efficiency while providing users with immediate feedback and reliable results.
