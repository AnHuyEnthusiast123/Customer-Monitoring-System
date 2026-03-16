package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/cenkalti/backoff/v4"
	"github.com/fsnotify/fsnotify"
)

var (
	facesDir    string
	featuresDir string
	backendURL  string
	maxRetries  uint64 = 5 // Default max retries
)

const (
	watchInterval = 5 * time.Second
)

type UploadData struct {
	FilePath string
	Metadata map[string]interface{} // Cập nhật: Sử dụng interface{} để linh hoạt (add visit_count)
}

func main() {
	flag.StringVar(&facesDir, "faces-dir", "/mnt/data/faces/", "Path to faces directory (PNG and TXT files)")
	flag.StringVar(&featuresDir, "features-dir", "/mnt/data/features/", "Path to features directory (BIN files)")
	flag.StringVar(&backendURL, "backend-url", "http://localhost:8080/api/upload", "Backend URL for uploading")
	flag.Uint64Var(&maxRetries, "max-retries", 5, "Max retries for upload")
	flag.Parse()

	if facesDir == "" || featuresDir == "" {
		log.Fatal("Error: Must provide -faces-dir and -features-dir")
	}

	uploadChan := make(chan UploadData, 200) // Tăng buffer
	go watchDirectories(uploadChan)
	go uploadWorker(uploadChan)
	select {}
}

func watchDirectories(uploadChan chan<- UploadData) {
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		log.Fatal(err)
	}
	defer watcher.Close()

	err = watcher.Add(facesDir)
	if err != nil {
		log.Printf("Error watching %s: %v", facesDir, err)
	}
	err = watcher.Add(featuresDir)
	if err != nil {
		log.Printf("Error watching %s: %v", featuresDir, err)
	}

	for {
		select {
		case event, ok := <-watcher.Events:
			if !ok {
				return
			}
			if event.Has(fsnotify.Create) || event.Has(fsnotify.Write) {
				ext := filepath.Ext(event.Name)
				if ext == ".png" || ext == ".txt" || ext == ".bin" {
					metadata := parseMetadata(event.Name)
					uploadChan <- UploadData{FilePath: event.Name, Metadata: metadata}
					log.Printf("Detected change: %s", event.Name)
				}
			}
		case err, ok := <-watcher.Errors:
			if !ok {
				return
			}
			log.Printf("Watch error: %v", err)
			time.Sleep(watchInterval)
		}
	}
}

func uploadWorker(uploadChan <-chan UploadData) {
	for data := range uploadChan {
		operation := func() error {
			return uploadFile(data)
		}
		b := backoff.NewExponentialBackOff()
		b.MaxElapsedTime = 5 * time.Minute
		err := backoff.Retry(operation, backoff.WithMaxRetries(b, maxRetries))
		if err != nil {
			log.Printf("Upload failed after retries for %s: %v", data.FilePath, err)
		} else {
			log.Printf("Uploaded: %s", data.FilePath)
			// os.Remove(data.FilePath) // Optional, enable if needed
		}
	}
}

func uploadFile(data UploadData) error {
	file, err := os.Open(data.FilePath)
	if err != nil {
		return err
	}
	defer file.Close()

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	part, err := writer.CreateFormFile("files", filepath.Base(data.FilePath))
	if err != nil {
		return err
	}
	_, err = io.Copy(part, file)
	if err != nil {
		return err
	}

	metadataJSON, _ := json.Marshal(data.Metadata)
	writer.WriteField("metadata", string(metadataJSON))

	err = writer.Close()
	if err != nil {
		return err
	}

	req, err := http.NewRequest("POST", backendURL, body)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}
	return nil
}

func parseMetadata(filePath string) map[string]interface{} {
	metadata := make(map[string]interface{})
	base := filepath.Base(filePath)
	ext := filepath.Ext(base)

	re := regexp.MustCompile(`person_(\d+)\.(png|txt|bin)`) // Update regex
	matches := re.FindStringSubmatch(base)
	if len(matches) < 2 {
		log.Printf("Warning: Failed to parse metadata from %s", base)
		return metadata
	}
	metadata["person_id"] = matches[1]

	// Nếu là .png hoặc .bin, tìm .txt tương ứng để đọc thêm
	if ext != ".txt" {
		txtPath := filepath.Join(filepath.Dir(filePath), "person_"+matches[1]+".txt")
		if _, err := os.Stat(txtPath); err == nil {
			filePath = txtPath // Chuyển sang đọc .txt
		}
	}

	// Đọc .txt để extract metadata
	if ext == ".txt" || filePath != "" { // Chỉ đọc nếu là .txt hoặc đã switch
		content, err := os.ReadFile(filePath)
		if err != nil {
			log.Printf("Warning: Failed to read %s: %v", filePath, err)
			return metadata
		}
		lines := strings.Split(string(content), "\n")
		if len(lines) > 0 {
			// Parse dòng 1: PersonID:1, TrackerUID:123, Count:(\d+), BBox:..., Quality:..., VisitCount:(\d+) (loại bỏ Timestamp)
			reLine1 := regexp.MustCompile(`PersonID:(\d+), TrackerUID:(\d+), Count:(\d+), BBox:([\d.]+),([\d.]+),([\d.]+),([\d.]+), Quality:([\d.]+), VisitCount:(\d+)`)
			m := reLine1.FindStringSubmatch(lines[0])
			if len(m) >= 10 {
				metadata["person_id"] = m[1] // Override nếu cần
				metadata["tracker_uid"] = m[2]
				metadata["counter"] = m[3]
				metadata["bbox_x1"] = m[4]
				metadata["bbox_y1"] = m[5]
				metadata["bbox_width"] = m[6]
				metadata["bbox_height"] = m[7]
				metadata["quality"] = m[8]
				metadata["visit_count"] = m[9] // THÊM: Parse visit_count
			}
			// Parse landmarks (dòng 2-6)
			for i := 1; i <= 5 && i < len(lines); i++ {
				reLandmark := regexp.MustCompile(`Landmark\d+: ([\d.]+), ([\d.]+)`)
				lm := reLandmark.FindStringSubmatch(lines[i])
				if len(lm) >= 3 {
					metadata[fmt.Sprintf("landmark%d_x", i)] = lm[1]
					metadata[fmt.Sprintf("landmark%d_y", i)] = lm[2]
				}
			}
		}
	}
	return metadata
}
