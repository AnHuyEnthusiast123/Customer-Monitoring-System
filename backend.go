package main

import (
	"database/sql"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v4"
	"github.com/gorilla/websocket"
	_ "github.com/mattn/go-sqlite3"
	"go.uber.org/zap"
	"os/signal"
)

type FaceData struct {
	PersonID    uint64 `json:"person_id"`
	VisitCount  int    `json:"visit_count"`
	FirstSeenTS uint64 `json:"first_seen_ts"`
	LastSeenTS  uint64 `json:"last_seen_ts"`
	Name        string `json:"name"`
}

var (
	db        *sql.DB
	mu        sync.Mutex
	uploadDir = "./uploads" // Changed to relative for user permissions
	clients   = make(map[*websocket.Conn]bool)
	broadcast = make(chan FaceData)
	rtspURL   = "rtsp://192.168.42.1/h264"
	logger    *zap.Logger
	jwtSecret = []byte(os.Getenv("JWT_SECRET")) // Set in env, e.g., export JWT_SECRET=secret
)

func main() {
	var err error
	logger, err = zap.NewProduction()
	if err != nil {
		log.Fatalf("Failed to initialize logger: %v", err)
	}
	defer logger.Sync()

	db, err = sql.Open("sqlite3", "./faces.db")
	if err != nil {
		logger.Fatal("Failed to open DB", zap.Error(err))
	}
	defer db.Close()

	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS face_data (
		person_id INTEGER PRIMARY KEY,
		visit_count INTEGER DEFAULT 1,
		first_seen_ts INTEGER,
		last_seen_ts INTEGER,
		name TEXT DEFAULT 'Unknown'
	)`)
	if err != nil {
		logger.Fatal("Failed to create table", zap.Error(err))
	}
	_, err = db.Exec("CREATE INDEX IF NOT EXISTS idx_ts ON face_data (first_seen_ts, last_seen_ts)")
	if err != nil {
		logger.Warn("Failed to create index", zap.Error(err))
	}

	if err := os.MkdirAll(uploadDir, os.ModePerm); err != nil {
		logger.Fatal("Failed to create upload dir", zap.Error(err))
	}

	r := gin.Default()
	gin.SetMode(gin.ReleaseMode)
	r.Use(cors.New(cors.Config{AllowOrigins: []string{"*"}, AllowHeaders: []string{"*"}, ExposeHeaders: []string{"Content-Length"}})) // Wildcard cho test
	// r.Use(authMiddleware()) // Commented out to avoid 401 errors for dev/testing

	r.POST("/api/upload", uploadHandler)
	r.GET("/api/stats", statsHandler)
	r.GET("/api/visits", visitsHandler)
	r.GET("/api/live", liveHandler)
	r.StaticFS("/uploads", http.Dir(uploadDir)) // Thay Static bằng StaticFS để full file server features
	r.Use(func(c *gin.Context) {
		if strings.HasPrefix(c.Request.URL.Path, "/uploads") {
			c.Header("Cache-Control", "no-cache, no-store, must-revalidate")
			c.Header("Pragma", "no-cache")
			c.Header("Expires", "0")
		}
		c.Next()
	})
	go handleMessages()
	go runMediaMTX()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigChan
		logger.Info("Shutting down...")
		db.Close()
		os.Exit(0)
	}()

	r.Run(":8080")
}

func authMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		tokenStr := c.GetHeader("Authorization")
		if tokenStr == "" {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "Unauthorized"})
			c.Abort()
			return
		}
		token, err := jwt.Parse(tokenStr, func(token *jwt.Token) (interface{}, error) {
			return jwtSecret, nil
		})
		if err != nil || !token.Valid {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid token"})
			c.Abort()
			return
		}
		c.Next()
	}
}

func uploadHandler(c *gin.Context) {
	form, err := c.MultipartForm()
	if err != nil {
		logger.Error("Multipart form error", zap.Error(err))
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	files := form.File["files"]
	for _, file := range files {
		if !strings.HasSuffix(file.Filename, ".png") && !strings.HasSuffix(file.Filename, ".txt") && !strings.HasSuffix(file.Filename, ".bin") {
			logger.Warn("Invalid file type", zap.String("filename", file.Filename))
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid file type"})
			return
		}
		sanitizedFilename := strings.Map(func(r rune) rune {
			if strings.IndexRune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-", r) >= 0 {
				return r
			}
			return -1
		}, file.Filename)
		savePath := filepath.Join(uploadDir, sanitizedFilename)
		if err := c.SaveUploadedFile(file, savePath); err != nil {
			logger.Error("Save file error", zap.Error(err))
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		const minPngSize = 50000 // 50KB threshold, adjust dựa test full PNG
		// Sau save .png
		if strings.HasSuffix(file.Filename, ".png") {
			info, err := os.Stat(savePath)
			if err != nil || info.Size() < minPngSize {
				logger.Error("Incomplete PNG", zap.String("file", file.Filename), zap.Int64("size", info.Size()))
				return
			}
		}
		// Nếu error, c.JSON(http.StatusBadRequest, gin.H{"error": "Incomplete file"})
	}

	metadataStr := form.Value["metadata"]
	if len(metadataStr) > 0 {
		var metadata map[string]interface{}
		if err := json.Unmarshal([]byte(metadataStr[0]), &metadata); err != nil {
			logger.Error("Unmarshal metadata error", zap.Error(err))
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		personIDStr, ok := metadata["person_id"].(string)
		if !ok {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid person_id"})
			return
		}
		personID, err := strconv.ParseUint(personIDStr, 10, 64)
		if err != nil || personID == 0 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid person_id"})
			return
		}
		loc, _ := time.LoadLocation("Asia/Ho_Chi_Minh")
		currentTS := uint64(time.Now().In(loc).UnixMilli())
		lastSeenTS := currentTS                                 // Default fallback
		if tsStr, ok := metadata["timestamp_us"].(string); ok { // THÊM: Sử dụng timestamp từ metadata nếu có
			ts, err := strconv.ParseUint(tsStr, 10, 64)
			if err == nil {
				lastSeenTS = ts
			}
		}

		visitCount := 1                                        // Default
		if vcStr, ok := metadata["visit_count"].(string); ok { // THÊM: Sử dụng visit_count từ metadata
			vc, err := strconv.Atoi(vcStr)
			if err == nil && vc > 0 {
				visitCount = vc
			}
		}

		mu.Lock()
		defer mu.Unlock()
		tx, err := db.Begin()
		if err != nil {
			logger.Error("Tx begin fail", zap.Error(err))
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		defer tx.Rollback()

		var existingVisitCount int
		var firstSeen, lastSeen uint64
		err = tx.QueryRow("SELECT visit_count, first_seen_ts, last_seen_ts FROM face_data WHERE person_id = ?", personID).Scan(&existingVisitCount, &firstSeen, &lastSeen)
		if err == sql.ErrNoRows {
			// Insert mới: Sử dụng visit_count từ metadata, lastSeenTS cho cả first và last
			_, err = tx.Exec("INSERT INTO face_data (person_id, visit_count, first_seen_ts, last_seen_ts) VALUES (?, ?, ?, ?)", personID, visitCount, lastSeenTS, lastSeenTS)
			if err != nil {
				logger.Error("DB insert error", zap.Error(err))
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
			existingVisitCount = visitCount
			firstSeen = lastSeenTS
			lastSeen = lastSeenTS
		} else if err == nil {
			// Update: Set visit_count từ metadata (sync), update last_seen_ts
			_, err = tx.Exec("UPDATE face_data SET visit_count = ?, last_seen_ts = ? WHERE person_id = ?", visitCount, lastSeenTS, personID)
			if err != nil {
				logger.Error("DB update error", zap.Error(err))
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
			existingVisitCount = visitCount
			lastSeen = lastSeenTS // Update lastSeen
		} else {
			logger.Error("DB query error", zap.Error(err))
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		if err := tx.Commit(); err != nil {
			logger.Error("Tx commit fail", zap.Error(err))
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		data := FaceData{PersonID: personID, VisitCount: existingVisitCount, FirstSeenTS: firstSeen, LastSeenTS: lastSeenTS}
		broadcast <- data
	}

	c.JSON(http.StatusOK, gin.H{"status": "uploaded", "files_received": len(files)})
}

func statsHandler(c *gin.Context) {
	mu.Lock()
	defer mu.Unlock()

	var loc, err = time.LoadLocation("Asia/Ho_Chi_Minh")
	if err != nil {
		logger.Error("Failed to load location", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to load timezone"})
		return
	}
	var totalPeople, totalVisits int
	db.QueryRow("SELECT COUNT(*), COALESCE(SUM(visit_count), 0) FROM face_data").Scan(&totalPeople, &totalVisits)

	now := time.Now().In(loc)
	todayStart := uint64(time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, loc).UnixMilli())
	var newToday, returningToday int
	db.QueryRow("SELECT COUNT(*) FROM face_data WHERE first_seen_ts >= ?", todayStart).Scan(&newToday)
	db.QueryRow("SELECT COUNT(*) FROM face_data WHERE visit_count > 1 AND last_seen_ts >= ?", todayStart).Scan(&returningToday)

	c.JSON(http.StatusOK, gin.H{
		"total_people":    totalPeople,
		"total_visits":    totalVisits,
		"new_today":       newToday,
		"returning_today": returningToday,
		"in_store":        0, // TODO
	})
}

func visitsHandler(c *gin.Context) {
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "50"))
	offset, _ := strconv.Atoi(c.DefaultQuery("offset", "0"))
	dateRange := c.Query("date_range")

	mu.Lock()
	defer mu.Unlock()

	query := "SELECT * FROM face_data"
	var args []interface{}
	if dateRange != "" {
		parts := strings.Split(dateRange, ":")
		if len(parts) == 2 {
			fromTS := parseTS(parts[0])
			toTS := parseTS(parts[1])
			query += " WHERE last_seen_ts BETWEEN ? AND ?"
			args = append(args, fromTS, toTS)
		}
	}
	query += " ORDER BY last_seen_ts DESC LIMIT ? OFFSET ?"
	args = append(args, limit, offset)

	rows, err := db.Query(query, args...)
	if err != nil {
		logger.Error("Visits query error", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer rows.Close()

	var allData []FaceData
	for rows.Next() {
		var data FaceData
		rows.Scan(&data.PersonID, &data.VisitCount, &data.FirstSeenTS, &data.LastSeenTS, &data.Name)
		allData = append(allData, data)
	}
	c.JSON(http.StatusOK, allData)
}

func parseTS(date string) uint64 {
	loc, _ := time.LoadLocation("Asia/Ho_Chi_Minh") // Ignore err for simplicity
	t, err := time.ParseInLocation("2006-01-02", date, loc)
	if err != nil {
		return 0 // Or handle error
	}
	return uint64(t.UnixMilli())
}

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return r.Header.Get("Origin") == "http://localhost:3000"
	},
}

func liveHandler(c *gin.Context) {
	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		logger.Error("WS upgrade error", zap.Error(err))
		return
	}
	defer conn.Close()

	mu.Lock()
	clients[conn] = true
	mu.Unlock()

	for {
		_, _, err := conn.ReadMessage()
		if err != nil {
			mu.Lock()
			delete(clients, conn)
			mu.Unlock()
			break
		}
	}
}

func handleMessages() {
	for {
		msg := <-broadcast
		mu.Lock()
		for client := range clients {
			err := client.WriteJSON(msg)
			if err != nil {
				logger.Error("WS write error", zap.Error(err))
				client.Close()
				delete(clients, client)
			}
		}
		mu.Unlock()
	}
}

func runMediaMTX() {
	cmd := exec.Command("./mediamtx", "/home/hyuuuan/awesomeProject/Uploader/config.yml ") // Hoặc path đến binary, với config.yml
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		logger.Error("MediaMTX start fail", zap.Error(err))
	} else {
		go func() {
			if err := cmd.Wait(); err != nil {
				logger.Error("MediaMTX exited with error", zap.Error(err))
			}
		}()
	}
}
