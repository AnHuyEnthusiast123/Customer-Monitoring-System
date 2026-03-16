# Customer-Monitoring-System

# Real-Time Face Recognition Pipeline

## Overview

This project develops an end-to-end system for embedded face recognition, focusing on real-time video processing, metadata handling, and a user-friendly dashboard for retail applications. It enables automatic customer tracking in stores, distinguishing new and returning visitors to provide business insights like total visits, unique customers, and daily new/returning ratios. The system is optimized for edge devices, ensuring low-latency operation without heavy cloud dependency.

<img width="768" height="164" alt="image" src="https://github.com/user-attachments/assets/e3abbd76-2a39-413b-ac67-37f998cf763e" />


Key features include:
- Face detection, quality assessment, feature extraction, and tracking on embedded hardware.
- File uploading with retry mechanisms for reliable data transfer.
- Backend server for data storage, API provision, and real-time WebSocket updates.
- Frontend dashboard for metrics visualization, charts, tables, and live streaming.

This aligns with the project summary in the CV: "Developed end-to-end system for embedded face detection and metadata handling as part of IoT security application," incorporating multi-threading in C for processing, Go for uploading and backend, and HTML/JS for the dashboard.

## Prerequisites

- **Hardware**:
  - Edge device (e.g., Milk-V Duo 256M board with CAM-GC2083 camera).
  - Host machine (e.g., PC for backend and frontend).

- **Software**:
  - Go (for backend and uploader).
  - C compiler and libraries (e.g., CVI_TDL for CV186X chip).
  - Node.js and npx (for serving frontend).
  - SQLite (for backend database).
  - MediaMTX (for RTSP to WebRTC streaming).
  - Models: scrfd_768_432_int8_1x.cvimodel, fqnet-v5_shufflenetv2-softmax.cvimodel, cviface-v6-s.cvimodel.

- **Environment**:
  - Embedded Linux on edge device (for running sample_vi_face_recog and uploader_riscv64).
  - Network connectivity between host and edge device (e.g., IP 192.168.42.1 for edge, 192.168.42.115 for host).

## Installation

1. Clone the repository:
   ```
   git clone [https://github.com/your-repo/awesomeProject.git](https://github.com/AnHuyEnthusiast123/Customer-Monitoring-System.git)
   ```

2. Build the Go binaries on the host:
   ```
   go build -o backend backend.go
   go build -o uploader main.go  # For host testing if needed
   go build -o uploader_riscv64 -tags riscv64 main.go  # Cross-compile for edge device
   ```

3. Transfer binaries to edge device:
   ```
   scp sample_vi_face_recog root@192.168.42.1:/root/
   scp uploader_riscv64 root@192.168.42.1:/root/
   ```

4. Place models in `/cvimodel/` on the edge device.

5. Install frontend dependencies (if any, though basic HTML/JS):
   ```
   cd Uploader/dashboard/
   # No npm install needed for pure HTML/JS; use npx for serving
   ```

## Usage

### On Host Machine

- Terminal 1 (Serve Frontend):
  ```
  cd awesomeProject/Uploader/dashboard/
  npx http-server -p 3000 --cors
  ```
  Access dashboard at `http://localhost:3000/index.html`.

- Terminal 2 (Build and Run Backend):
  ```
  export GIN_MODE=release
  ./backend
  ```
  Backend runs on port 8080 (e.g., APIs at `/api/upload`, WebSocket at `/api/live`).

### On Edge Device

- Terminal 1 (Run Video Processing):
  ```
  ./sample_vi_face_recog cvimodel/scrfd_768_432_int8_1x.cvimodel cvimodel/fqnet-v5_shufflenetv2-softmax.cvimodel cvimodel/cviface-v6-s.cvimodel RGB888
  ```
  Processes camera input, generates .png/.txt/.bin files in `/mnt/data/faces/` and `/mnt/data/features/`.

- Terminal 2 (Run Uploader):
  ```
  ./uploader_riscv64 -faces-dir=/mnt/data/faces -features-dir=/mnt/data/features -backend-url=http://192.168.42.115:8080/api/upload
  ```
  Watches directories and uploads to backend (replace IP with host's from `ip a`).

## Architecture and Workflow

The system follows an end-to-end pipeline:

<img width="523" height="780" alt="image" src="https://github.com/user-attachments/assets/8627a59b-c0e7-417a-b024-0c52313b1b1f" />


1. **Video Processing & Detection (C - sample_vi_face_recog.c)**:

  <img width="557" height="768" alt="image" src="https://github.com/user-attachments/assets/50e06c00-5e8e-491d-bda8-948de871ae3a" />

   - Logic: Multithreaded processing of video from camera, detect faces, and generate files.
   - Input: Video frames from camera (RTSP or VI pipeline on CV186X chip, configured via SAMPLE_TDL_MW_CONFIG_S).
   - Steps:
     - Initialize system: Configure VI/VPSS/VENC/RTSP, load models (RetinaFace for detection, FaceQuality for quality, FaceRecognition for features).
     - Threads:
       - tdl_thread: Detect faces (CVI_TDL_ScrFDFace), assess quality (CVI_TDL_FaceQuality), extract features (CVI_TDL_FaceRecognition), track (CVI_TDL_DeepSORT).
       - Filter: Process only faces with quality >0.5 and stable tracker state.
       - Crop image, enqueue data to buffer (io_data_t with u_id, image, feature, bbox, pts).
       - image_writer_thread: Dequeue, save .png (stbi_write_png), .bin (feature vector), .txt (metadata: u_id, count, bbox, quality, visit_count=1, landmarks).
       - venc_thread: Encode frames and stream RTSP.
     - Error handling: CHECK_ERROR macro, signal handler for shutdown.
     - Cleanup: Free resources, destroy handles.
   - Output: Files in /mnt/data/faces/ and /features/.
  
    

3. **Upload Files & Metadata (Go - main.go)**:

   <img width="880" height="860" alt="image" src="https://github.com/user-attachments/assets/368e9408-26c5-4ddc-b151-0d402ecc449f" />

   - Logic: Watch directories and upload files/metadata to backend.
   - Input: Files from /mnt/data/faces/ (.png, .txt) and /features/ (.bin).
   - Steps:
     - Parse args: facesDir, featuresDir, backendURL, maxRetries.
     - Watch directories with fsnotify (Create/Write events).
     - On new file: Parse metadata (regex from filename for person_id, read .txt for bbox/quality/landmarks/visit_count).
     - Enqueue UploadData to channel (buffer 200).
     - Worker: Upload multipart form (files + JSON metadata) via HTTP POST, with exponential backoff retry (max 5, up to 5 mins).
     - Files not deleted after upload (commented).
   - Output: Data sent to backend /api/upload.

5. **Backend Server & Database (Go - backend.go)**:

   <img width="960" height="565" alt="image" src="https://github.com/user-attachments/assets/a189d9a4-e7bd-43a5-8927-3dc60dd7229a" />

   - Logic: Handle uploads, update DB, provide APIs/WS, stream live.
   - Input: Uploads from uploader (multipart files + JSON metadata).
   - Steps:
     - Init: Open SQLite DB, create table (person_id, visit_count, first/last_seen_ts, name), index on ts.
     - Routes (Gin):
       - POST /api/upload: Save files to ./uploads/ (sanitize filename), parse metadata (person_id, visit_count, timestamp fallback current ms), DB transaction (insert new with first=last=current, update visit_count/last_seen if existing), broadcast FaceData via WS channel.
       - GET /api/stats: Lock mu, query COUNT/SUM for total_people/visits, new_today (first >= todayStart), returning_today (last >= todayStart AND first < todayStart).
       - GET /api/visits: Query all or filter date_range, return []FaceData.
       - GET /api/live: Upgrade to WS, add client, handle messages (read to keep alive).
     - WS: handleMessages goroutine sends JSON to all clients from broadcast channel.
     - MediaMTX: Run subprocess for RTSP stream (hardcoded URL).
     - CORS/Auth: For localhost:3000, JWT middleware (commented).
   - Output: DB updates, WS broadcasts, API responses, static /uploads, MediaMTX stream.

7. **Frontend Dashboard (HTML/JS - index.html)**:

   <img width="960" height="311" alt="image" src="https://github.com/user-attachments/assets/bed441bf-2ac6-42a8-8c24-faac92f87ec7" />

   - Logic: Display data from APIs/WS, charts/table, live stream.
   - Input: APIs from backend (/api/stats, /api/visits), WS updates, images from /uploads, stream from MediaMTX.
   - Steps:
     - Config apiBase from query param.
     - WS connect: onmessage reload metrics/charts/table.
     - loadStats: Fetch /api/stats, render metric cards (total_people/visits, new/returning today).
     - loadVisits: Fetch /api/visits, render charts (pie: new/returning ratio, line: daily visits, bar: top returning).
     - DataTable: Ajax /api/visits with date_range filter (Flatpickr), columns person_id/ts/visit_count/image/status.
     - Live Stream: WebRTC setup (RTCPeerConnection, offer/answer via POST to MediaMTX WHEP, trickle ICE PATCH).
   - Output: UI with cards/charts/table/stream.
  
  ## Result 
  ### Board Processing
  <img width="346" height="364" alt="image" src="https://github.com/user-attachments/assets/0e38a5d9-7e5a-4f1f-9143-59e565e9005f" />

  ### UI
  <img width="662" height="553" alt="image" src="https://github.com/user-attachments/assets/5944aec4-0399-4235-b8a6-0be18029ba71" />

