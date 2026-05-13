#define LOG_TAG "SampleFD"
#define LOG_LEVEL LOG_LEVEL_INFO

#include "middleware_utils.h"
#include "sample_utils.h"
#include "vi_vo_utils.h"

#include <core/utils/vpss_helper.h>
#include <cvi_comm.h>
#include <rtsp.h>
#include <sample_comm.h>
#include "cvi_tdl.h"

#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <inttypes.h>
#include <math.h>
#include <dirent.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#ifndef STBI_rgb
#define STBI_rgb 3
#endif

// Constants
#define OUTPUT_BUFFER_SIZE 1000
#define MAX_FILENAME_LEN 64
#define BUFFER_RETRY_DELAY_US 100000  // 100 ms
#define VPSS_FRAME_TIMEOUT_MS 2000
#define TDL_VPSS_TIMEOUT_MS 1000
#define MAX_VENC_STOP_RETRIES 5
#define VENC_STOP_RETRY_DELAY_US 20000
#define PROCESSED_UID_INIT_CAPACITY 16
#define TRACKED_ID_INIT_CAPACITY 16
#define HEAD_POSE_QUALITY_THRESHOLD 0.3f
#define MIN_COLOR_VALUE 64
#define MAX_COLOR_VALUE 256
#define IO_THREAD_EMPTY_DELAY_US 100000  // 100 ms
#define FEATURE_DIR "/mnt/data/features/"  // Thư mục riêng cho .bin
#define MAX_DB_SIZE 1000  // Kích thước tối đa DB, có thể tăng nếu cần
#define COSINE_THRESHOLD 0.6f  // Threshold mặc định, sẽ config sau
#define DB_RELOAD_INTERVAL_SEC 60  // Interval reload DB

// Structures
typedef struct {
    uint64_t u_id;
    float quality;  // Face quality
    cvtdl_image_t image;
    uint32_t counter;
    cvtdl_bbox_t bbox;
    cvtdl_pts_t pts;  // Store 5 facial landmarks
    cvtdl_feature_t feature;
    uint64_t timestamp_us; // THÊM: Thời gian (micro giây)
    uint64_t person_id;
} io_data_t;

typedef struct {
    SAMPLE_TDL_MW_CONTEXT *pstMWContext;
    cvitdl_service_handle_t stServiceHandle;
} venc_thread_arg_t;

typedef struct {
    cvitdl_handle_t stTDLHandle;
    bool bTrackingWithFeature;
} tdl_thread_arg_t;

typedef struct {
    uint64_t u_id;
    int state;  // CVI_TRACKER_NEW, CVI_TRACKER_UNSTABLE, CVI_TRACKER_STABLE
} tracked_id_t;

typedef struct {
    uint64_t person_id;        // THÊM: ID cố định cho từng người (khác với tracker UID)
    cvtdl_feature_t feature;
    char name[64];
    uint32_t visit_count;      // <-- Đây chính là total_visit_person
    uint64_t last_seen_ts;     // Để debug (optional)
} face_db_t;

// Globals
static volatile bool g_bExit = false;
static volatile bool g_bRunImageWriter = true;
static volatile uint32_t g_face_counter = 0;  // Track total faces processed

static cvtdl_face_t g_stFaceMeta = {0};
static cvtdl_tracker_t g_stTrackerMeta = {0};
static io_data_t g_data_buffer[OUTPUT_BUFFER_SIZE];
static int g_rear_idx = 0;
static int g_front_idx = 0;

// global DB
static face_db_t g_face_db[MAX_DB_SIZE];
static size_t g_db_count = 0;
static pthread_mutex_t g_db_mutex = PTHREAD_MUTEX_INITIALIZER;  // Để lock khi access DB
static uint64_t g_next_person_id = 1;  // Bắt đầu từ 1, tăng dần mỗi người mới
static volatile uint32_t g_total_people = 0;     // Tổng số người khác nhau từng xuất hiện
static volatile uint32_t g_total_visits_all = 0; // Tổng tất cả các lượt ghé thăm (nếu cần)

// Mutexes
static pthread_mutex_t g_result_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_io_mutex = PTHREAD_MUTEX_INITIALIZER;

// Forward declarations
static CVI_S32 handle_error(CVI_S32 ret, const char *func, int line, const char *msg);
static void set_sample_mot_config(cvtdl_deepsort_config_t *ds_conf);
static cvtdl_service_brush_t get_random_brush(uint64_t seed, int min);
static void *image_writer_thread(void *args);
static void *venc_thread(void *args);
static void *tdl_thread(void *args);
static void signal_handler(CVI_S32 signo);
static CVI_S32 init_system(const char *retina_model_path, const char *quality_model_path, const char *recognition_model_path, PIXEL_FORMAT_E enInputFormat, SAMPLE_TDL_MW_CONFIG_S *stMWConfig);
static CVI_S32 init_tdl(cvitdl_handle_t *stTDLHandle, cvitdl_service_handle_t *stServiceHandle, const char *retina_model_path, const char *quality_model_path, const char *recognition_model_path);
static void cleanup_system(SAMPLE_TDL_MW_CONTEXT *stMWContext, cvitdl_handle_t stTDLHandle, cvitdl_service_handle_t stServiceHandle);
static CVI_S32 process_frame_for_venc(const venc_thread_arg_t *args, VIDEO_FRAME_INFO_S *stFrame);
static CVI_S32 detect_and_track(const tdl_thread_arg_t *args, VIDEO_FRAME_INFO_S *stFrame, cvtdl_face_t *stFaceMeta, cvtdl_tracker_t *stTrackerMeta);
static bool is_buffer_empty(void);
static bool is_buffer_full(void);
static int get_next_rear_idx(void);
static int get_next_front_idx(void);
static void enqueue_io_data(const io_data_t *data);
static bool dequeue_io_data(io_data_t *data);
static void write_face_image_and_metadata(const io_data_t *data);
static void log_tracker_changes(const cvtdl_face_t *stFaceMeta, const cvtdl_tracker_t *stTrackerMeta, const tracked_id_t *prev_tracked_ids, size_t prev_count);
static bool is_uid_processed(uint64_t uid, const uint64_t *processed_uids, size_t count);
static void add_processed_uid(uint64_t uid, uint64_t **processed_uids, size_t *count, size_t *capacity);
// Cập nhật chữ ký hàm để nhận timestamp
static CVI_S32 crop_and_enqueue_face(VIDEO_FRAME_INFO_S *stFrame, const cvtdl_face_info_t *face_info, const cvtdl_tracker_info_t *tracker_info, uint64_t timestamp_us); 

// Macro for error checking
#define CHECK_ERROR(ret, msg) handle_error(ret, __func__, __LINE__, msg)

static CVI_S32 handle_error(CVI_S32 ret, const char *func, int line, const char *msg) {
    if (ret != CVI_SUCCESS) {
        printf("[%s] ERROR: %s at %s:%d failed with %#x\n", LOG_TAG, msg, func, line, ret);
        g_bExit = true;
    }
    return ret;
}

float cosine_similarity(const cvtdl_feature_t *feat1, const cvtdl_feature_t *feat2) {
    if (feat1->size != feat2->size || feat1->size == 0 || feat1->type != feat2->type) {
        printf("[%s] WARNING: Feature mismatch - size1=%u, size2=%u, type1=%d, type2=%d\n",
               LOG_TAG, feat1->size, feat2->size, feat1->type, feat2->type);
        return 0.0f;
    }

    if (feat1->type == TYPE_INT8) {
        int32_t dot_product = 0;
        int32_t norm1 = 0;
        int32_t norm2 = 0;
        int8_t *ptr1 = (int8_t *)feat1->ptr;
        int8_t *ptr2 = (int8_t *)feat2->ptr;
        for (uint32_t i = 0; i < feat1->size; i++) {
            dot_product += (int32_t)ptr1[i] * ptr2[i];
            norm1 += (int32_t)ptr1[i] * ptr1[i];
            norm2 += (int32_t)ptr2[i] * ptr2[i];
        }
        float fnorm1 = sqrtf((float)norm1);
        float fnorm2 = sqrtf((float)norm2);
        if (fnorm1 == 0.0f || fnorm2 == 0.0f) {
            printf("[%s] WARNING: Zero norm in int8 cosine\n", LOG_TAG);
            return 0.0f;
        }
        return (float)dot_product / (fnorm1 * fnorm2);
    } else if (feat1->type == TYPE_FLOAT) {
        float dot_product = 0.0f;
        float norm1 = 0.0f;
        float norm2 = 0.0f;
        float *ptr1 = (float *)feat1->ptr;
        float *ptr2 = (float *)feat2->ptr;
        for (uint32_t i = 0; i < feat1->size; i++) {
            dot_product += ptr1[i] * ptr2[i];
            norm1 += powf(ptr1[i], 2);  // Sửa dùng powf cho float
            norm2 += powf(ptr2[i], 2);
        }
        norm1 = sqrtf(norm1);
        norm2 = sqrtf(norm2);
        if (norm1 == 0.0f || norm2 == 0.0f) {
            printf("[%s] WARNING: Zero norm in float cosine\n", LOG_TAG);
            return 0.0f;
        }
        return dot_product / (norm1 * norm2);
    } else {
        printf("[%s] ERROR: Unsupported feature type=%d\n", LOG_TAG, feat1->type);
        return 0.0f;
    }
}

static CVI_S32 load_face_db(void) {
    // THÊM: Reset globals trước khi load
    g_total_people = 0;
    g_total_visits_all = 0;
    g_next_person_id = 1;  // Reset để tính max từ loaded IDs

    DIR *dir = opendir(FEATURE_DIR);
    if (!dir) {
        printf("[%s] ERROR: Failed to open directory %s\n", LOG_TAG, FEATURE_DIR);
        return CVI_FAILURE;
    }
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strstr(entry->d_name, ".bin") == NULL) continue; // Chỉ load .bin
        char bin_path[MAX_FILENAME_LEN];
        snprintf(bin_path, MAX_FILENAME_LEN, "%s%s", FEATURE_DIR, entry->d_name);
        FILE *bin_file = fopen(bin_path, "rb");
        if (!bin_file) continue;
        uint32_t vec_size;
        fread(&vec_size, sizeof(uint32_t), 1, bin_file);
        if (g_db_count >= MAX_DB_SIZE) {
            printf("[%s] WARNING: DB full, skipping %s\n", LOG_TAG, bin_path);
            fclose(bin_file);
            continue;
        }
        g_face_db[g_db_count].feature.size = vec_size;
        g_face_db[g_db_count].feature.type = TYPE_INT8;  // Giả sử model output int8, sửa nếu khác
        g_face_db[g_db_count].feature.ptr = malloc(sizeof(int8_t) * vec_size);
        if (!g_face_db[g_db_count].feature.ptr) {
            fclose(bin_file);
            continue;
        }
        fread(g_face_db[g_db_count].feature.ptr, sizeof(int8_t), vec_size, bin_file);
        fclose(bin_file);
        // Parse person_id từ filename
        sscanf(entry->d_name, "person_%" PRIu64 ".bin", &g_face_db[g_db_count].person_id);
        strcpy(g_face_db[g_db_count].name, "Unknown");
        g_face_db[g_db_count].last_seen_ts = 0;  // Init 0, có thể load nếu thêm vào .txt sau

        // THÊM: Load visit_count từ .txt tương ứng
        char txt_path[MAX_FILENAME_LEN];
        snprintf(txt_path, MAX_FILENAME_LEN, "/mnt/data/faces/person_%" PRIu64 ".txt", g_face_db[g_db_count].person_id);
        FILE *txt_file = fopen(txt_path, "r");
        if (txt_file) {
            char line[512];
            if (fgets(line, sizeof(line), txt_file)) {
                char *pos = strstr(line, "VisitCount:");
                if (pos) {
                    uint32_t visit;
                    sscanf(pos, "VisitCount:%u", &visit);
                    g_face_db[g_db_count].visit_count = visit;
                    printf("[%s] INFO: Loaded visit_count=%u for Person ID %" PRIu64 "\n", LOG_TAG, visit, g_face_db[g_db_count].person_id);
                } else {
                    g_face_db[g_db_count].visit_count = 1;  // Default nếu không parse được
                }
            }
            fclose(txt_file);
        } else {
            printf("[%s] WARNING: No .txt for Person ID %" PRIu64 ", init visit_count=1\n", LOG_TAG, g_face_db[g_db_count].person_id);
            g_face_db[g_db_count].visit_count = 1;
        }

        // THÊM: Cập nhật totals
        g_total_visits_all += g_face_db[g_db_count].visit_count;
        g_total_people++;

        // Kiểm tra duplicate (giữ nguyên)
        bool duplicate = false;
        for (size_t i = 0; i < g_db_count; i++) {
            if (g_face_db[i].person_id == g_face_db[g_db_count].person_id) {
                duplicate = true;
                printf("[%s] WARNING: Duplicate .bin for Person ID %" PRIu64 ", skipping\n", LOG_TAG, g_face_db[g_db_count].person_id);
                free(g_face_db[g_db_count].feature.ptr);
                break;
            }
        }
        if (duplicate) continue;

        // Cập nhật g_next_person_id (giữ nguyên)
        if (g_face_db[g_db_count].person_id >= g_next_person_id) {
            g_next_person_id = g_face_db[g_db_count].person_id + 1;
        }
        g_db_count++;
    }
    closedir(dir);
    printf("[%s] INFO: Loaded %zu entries from DB\n", LOG_TAG, g_db_count);
    return CVI_SUCCESS;
}

static void *db_reload_thread(void *args) {
    while (!g_bExit) {
        sleep(DB_RELOAD_INTERVAL_SEC);
        pthread_mutex_lock(&g_db_mutex);
        for (size_t i = 0; i < g_db_count; i++) {
            if (g_face_db[i].feature.ptr) free(g_face_db[i].feature.ptr);
        }
        g_db_count = 0;
        
        g_total_people = 0;
        g_total_visits_all = 0;
        g_next_person_id = 1;
        load_face_db(); 
        pthread_mutex_unlock(&g_db_mutex);
    }
    pthread_exit(NULL);
}



void set_sample_mot_config(cvtdl_deepsort_config_t *ds_conf) {
    ds_conf->ktracker_conf.max_unmatched_num = 10;
    ds_conf->ktracker_conf.accreditation_threshold = 10;
    ds_conf->ktracker_conf.P_beta[2] = 0.1;
    ds_conf->ktracker_conf.P_beta[6] = 2.5e-2;
    ds_conf->kfilter_conf.Q_beta[2] = 0.1;
    ds_conf->kfilter_conf.Q_beta[6] = 2.5e-2;
}

cvtdl_service_brush_t get_random_brush(uint64_t seed, int min) {
    float scale = (256. - (float)min) / 256.;
    srand((uint32_t)seed);
    cvtdl_service_brush_t brush = {
        .color.r = (int)((floor(((float)rand() / (RAND_MAX)) * 256.)) * scale) + min,
        .color.g = (int)((floor(((float)rand() / (RAND_MAX)) * 256.)) * scale) + min,
        .color.b = (int)((floor(((float)rand() / (RAND_MAX)) * 256.)) * scale) + min,
        .size = 2,
    };
    return brush;
}

static void *image_writer_thread(void *args) {
    (void)args;  // Unused
    printf("[%s] INFO: Image writer thread started\n", LOG_TAG);

    while (g_bRunImageWriter && !g_bExit) {
        if (is_buffer_empty()) {
            usleep(IO_THREAD_EMPTY_DELAY_US);
            continue;
        }

        io_data_t data;
        if (!dequeue_io_data(&data)) {
            continue;
        }

        write_face_image_and_metadata(&data);

        // Free the image and feature data
        CVI_TDL_Free(&data.image);
        if (data.feature.ptr) {
            free(data.feature.ptr);
        }
    }

    // Cleanup remaining buffer
    printf("[%s] INFO: Freeing remaining buffer data...\n", LOG_TAG);
    while (!is_buffer_empty()) {
        io_data_t remaining_data;
        if (dequeue_io_data(&remaining_data)) {
            CVI_TDL_Free(&remaining_data.image);
            if (remaining_data.feature.ptr) {
                free(remaining_data.feature.ptr);
            }
        }
    }

    printf("[%s] INFO: Cleaning up image writer resources\n", LOG_TAG);
    pthread_mutex_destroy(&g_io_mutex);
    pthread_exit(NULL);
}

static void write_face_image_and_metadata(const io_data_t *data) {
    char png_filename[MAX_FILENAME_LEN];
    char txt_filename[MAX_FILENAME_LEN];
    char bin_filename[MAX_FILENAME_LEN];
    // THAY ĐỔI: Sử dụng person_id và visit1 (vì chỉ lưu lần đầu)
    snprintf(png_filename, MAX_FILENAME_LEN, "/mnt/data/faces/person_%" PRIu64 ".png", data->person_id);
    snprintf(txt_filename, MAX_FILENAME_LEN, "/mnt/data/faces/person_%" PRIu64 ".txt", data->person_id);
    snprintf(bin_filename, MAX_FILENAME_LEN, "%sperson_%" PRIu64 ".bin", FEATURE_DIR, data->person_id);
    // Viết metadata vào TXT (Chỉ bbox, quality, pts)
    FILE *txt_file = fopen(txt_filename, "w");
    if (!txt_file) {
        printf("[%s] ERROR: Failed to open %s for writing\n", LOG_TAG, txt_filename);
        return;
    }
    // Dòng 1: Person ID, Tracker UID, Timestamp, Counter, BBox, Quality (THAY ĐỔI: Thêm person_id)
    fprintf(txt_file, "PersonID:%" PRIu64 ", TrackerUID:%" PRIu64 ", Count:%u, BBox:%.2f,%.2f,%.2f,%.2f, Quality:%.2f, VisitCount:%u\n",
        data->person_id, data->u_id, data->counter,
        data->bbox.x1, data->bbox.y1,
        data->bbox.x2 - data->bbox.x1, data->bbox.y2 - data->bbox.y1,
        data->quality, g_face_db[g_db_count - 1].visit_count); 
    // Dòng 2-6: 5 landmarks (pts)
    for (int i = 0; i < 5; i++) {
        fprintf(txt_file, "Landmark%d: %.2f, %.2f\n", i + 1, data->pts.x[i], data->pts.y[i]);
    }
    fflush(txt_file);
    fclose(txt_file);
    // Viết feature vector vào BIN
    if (data->feature.size > 0 && data->feature.ptr) {
        FILE *bin_file = fopen(bin_filename, "wb");
        if (!bin_file) {
            printf("[%s] ERROR: Failed to open %s for writing\n", LOG_TAG, bin_filename);
            return;
        }
        printf("[%s] DEBUG: Writing .bin - feature type=%d, size=%u\n", LOG_TAG, data->feature.type, data->feature.size);
        uint32_t vec_size = data->feature.size;
        fwrite(&vec_size, sizeof(uint32_t), 1, bin_file);
        size_t elem_size = (data->feature.type == TYPE_INT8) ? sizeof(int8_t) : sizeof(float);
        fwrite(data->feature.ptr, elem_size, data->feature.size, bin_file);
        printf("[%s] INFO: Writing feature vector size: %u elements (type=%d) to %s\n", LOG_TAG, vec_size, data->feature.type, bin_filename);
        fclose(bin_file);
    }
    // Write image to PNG
    if (data->image.pix_format != PIXEL_FORMAT_RGB_888) {
        printf("[%s] WARNING: Unsupported image format: %d\n", LOG_TAG, data->image.pix_format);
    } else if (data->image.width == 0) {
        printf("[%s] WARNING: Empty target image\n", LOG_TAG);
    } else {
        printf("[%s] INFO: Writing face image: %s\n", LOG_TAG, png_filename);
        if (!stbi_write_png(png_filename, data->image.width, data->image.height, STBI_rgb,
                            data->image.pix[0], data->image.stride[0])) {
            printf("[%s] ERROR: Failed to write %s\n", LOG_TAG, png_filename);
        }
    }
}

static bool is_buffer_empty(void) {
    bool empty;
    pthread_mutex_lock(&g_io_mutex);
    empty = (g_front_idx == g_rear_idx);
    pthread_mutex_unlock(&g_io_mutex);
    return empty;
}

static bool is_buffer_full(void) {
    int next_rear = get_next_rear_idx();
    bool full;
    pthread_mutex_lock(&g_io_mutex);
    full = (next_rear == g_front_idx);
    pthread_mutex_unlock(&g_io_mutex);
    return full;
}

static int get_next_rear_idx(void) {
    return (g_rear_idx + 1) % OUTPUT_BUFFER_SIZE;
}

static int get_next_front_idx(void) {
    return (g_front_idx + 1) % OUTPUT_BUFFER_SIZE;
}

static void enqueue_io_data(const io_data_t *data) {
    int target_idx = get_next_rear_idx();
    if (is_buffer_full()) {
        printf("[%s] WARNING: Buffer full, dropping data!\n", LOG_TAG);
        return;
    }

    g_data_buffer[target_idx] = *data;  // Shallow copy; assume image is already copied
    pthread_mutex_lock(&g_io_mutex);
    g_rear_idx = target_idx;
    pthread_mutex_unlock(&g_io_mutex);
}

static bool dequeue_io_data(io_data_t *data) {
    int target_idx = get_next_front_idx();
    pthread_mutex_lock(&g_io_mutex);
    if (g_front_idx == g_rear_idx) {
        pthread_mutex_unlock(&g_io_mutex);
        return false;
    }
    *data = g_data_buffer[target_idx];
    g_front_idx = target_idx;
    pthread_mutex_unlock(&g_io_mutex);
    return true;
}

static void *venc_thread(void *args) {
    printf("[%s] INFO: VENC thread started\n", LOG_TAG);
    const venc_thread_arg_t *venc_args = (const venc_thread_arg_t *)args;
    VIDEO_FRAME_INFO_S stFrame = {0};

    while (!g_bExit) {
        CVI_S32 s32Ret = CVI_VPSS_GetChnFrame(0, 0, &stFrame, VPSS_FRAME_TIMEOUT_MS);
        if (s32Ret != CVI_SUCCESS) {
            CHECK_ERROR(s32Ret, "CVI_VPSS_GetChnFrame chn0");
            break;
        }

        if (process_frame_for_venc(venc_args, &stFrame) != CVI_SUCCESS) {
            CVI_VPSS_ReleaseChnFrame(0, 0, &stFrame);
            break;
        }

        CVI_VPSS_ReleaseChnFrame(0, 0, &stFrame);
    }

    // Cleanup VENC
    printf("[%s] INFO: Cleaning up VENC resources\n", LOG_TAG);
    MMF_CHN_S src_chn = { .enModId = CVI_ID_VPSS, .s32DevId = 0, .s32ChnId = 0 };
    MMF_CHN_S dest_chn = { .enModId = CVI_ID_VENC, .s32DevId = 0, .s32ChnId = 0 };
    CVI_SYS_UnBind(&src_chn, &dest_chn);

    int retry = 0;
    CVI_S32 s32Ret;
    do {
        s32Ret = CVI_VENC_StopRecvFrame(0);
        if (s32Ret == CVI_SUCCESS) break;
        printf("[%s] ERROR: CVI_VENC_StopRecvFrame failed (retry %d/%d)\n", LOG_TAG, retry + 1, MAX_VENC_STOP_RETRIES);
        usleep(VENC_STOP_RETRY_DELAY_US);
    } while (++retry < MAX_VENC_STOP_RETRIES);
    CHECK_ERROR(s32Ret, "CVI_VENC_StopRecvFrame");

    CVI_VENC_DestroyChn(0);

    printf("[%s] INFO: VENC thread exited\n", LOG_TAG);
    pthread_exit(NULL);
}

static CVI_S32 process_frame_for_venc(const venc_thread_arg_t *args, VIDEO_FRAME_INFO_S *stFrame) {
    cvtdl_face_t stFaceMeta = {0};
    cvtdl_tracker_t stTrackerMeta = {0};
    cvtdl_service_brush_t *brushes = NULL;
    pthread_mutex_lock(&g_result_mutex);
    CVI_TDL_CopyFaceMeta(&g_stFaceMeta, &stFaceMeta);
    CVI_TDL_CopyTrackerMeta(&g_stTrackerMeta, &stTrackerMeta);
    pthread_mutex_unlock(&g_result_mutex);
    CVI_S32 s32Ret = CVI_SUCCESS;
    if (stFaceMeta.size > 0) {
        brushes = calloc(stFaceMeta.size, sizeof(cvtdl_service_brush_t));
        if (!brushes) {
            CVI_TDL_Free(&stFaceMeta);
            CVI_TDL_Free(&stTrackerMeta);
            return CVI_FAILURE;
        }
        cvtdl_service_brush_t grey_brush = CVI_TDL_Service_GetDefaultBrush();
        grey_brush.color.r = grey_brush.color.g = grey_brush.color.b = 105;
        cvtdl_service_brush_t green_brush = CVI_TDL_Service_GetDefaultBrush();
        green_brush.color.r = 0;
        green_brush.color.g = 255;
        green_brush.color.b = 0;
        for (uint32_t i = 0; i < stFaceMeta.size; i++) {
            int state = stTrackerMeta.info[i].state;
            brushes[i] = (state == CVI_TRACKER_NEW) ? green_brush :
                             (state == CVI_TRACKER_UNSTABLE) ? grey_brush :
            get_random_brush(stFaceMeta.info[i].unique_id, MIN_COLOR_VALUE);
        }
        // Thêm phần cập nhật name với similarity
        for (uint32_t i = 0; i < stFaceMeta.size; i++) {
            pthread_mutex_lock(&g_db_mutex);
            float max_sim = 0.0f;
            int matched_idx = -1;
            for (size_t j = 0; j < g_db_count; j++) {
                float sim = cosine_similarity(&stFaceMeta.info[i].feature, &g_face_db[j].feature);  // Fix typo: info[i] thay vì info[j]
                if (sim > max_sim) {
                    max_sim = sim;
                    matched_idx = (int)j;
                }
            }
            pthread_mutex_unlock(&g_db_mutex);

            if (matched_idx != -1 && max_sim > COSINE_THRESHOLD) {
                // Cá nhân hóa cho khách cũ/mới đã match
                snprintf(stFaceMeta.info[i].name, sizeof(stFaceMeta.info[i].name),
                        "Guest_ID %" PRIu64 " visited %u times",
                        g_face_db[matched_idx].person_id, g_face_db[matched_idx].visit_count);
            }
        }
        s32Ret = CVI_TDL_Service_FaceDrawRect2(args->stServiceHandle, (cvtdl_face_t*)&stFaceMeta, stFrame, true, brushes);
        CHECK_ERROR(s32Ret, "CVI_TDL_Service_FaceDrawRect2");
        if (s32Ret != CVI_SUCCESS) goto cleanup;
        s32Ret = CVI_TDL_Service_FaceDrawRect(args->stServiceHandle, (cvtdl_face_t*)&stFaceMeta, stFrame, true, CVI_TDL_Service_GetDefaultBrush());
        CHECK_ERROR(s32Ret, "CVI_TDL_Service_FaceDrawRect");
        if (s32Ret != CVI_SUCCESS) goto cleanup;
        s32Ret = CVI_TDL_Service_FaceDraw5Landmark((cvtdl_face_t*)&stFaceMeta, stFrame);
        CHECK_ERROR(s32Ret, "CVI_TDL_Service_FaceDraw5Landmark");
        if (s32Ret != CVI_SUCCESS) goto cleanup;
    }
    s32Ret = SAMPLE_TDL_Send_Frame_RTSP(stFrame, args->pstMWContext);
cleanup:
    if (brushes) {
        free(brushes);
    }
    CVI_TDL_Free(&stFaceMeta);
    CVI_TDL_Free(&stTrackerMeta);
    return s32Ret;
}

static CVI_S32 detect_and_track(const tdl_thread_arg_t *args, VIDEO_FRAME_INFO_S *stFrame, cvtdl_face_t *stFaceMeta, cvtdl_tracker_t *stTrackerMeta) {
    CVI_S32 s32Ret = CVI_TDL_ScrFDFace(args->stTDLHandle, stFrame, stFaceMeta);
    CHECK_ERROR(s32Ret, "CVI_TDL_ScrFDFace");
    if (s32Ret != CVI_SUCCESS) return s32Ret;

    s32Ret = CVI_TDL_Service_FaceAngleForAll(stFaceMeta);
    CHECK_ERROR(s32Ret, "CVI_TDL_Service_FaceAngleForAll");
    if (s32Ret != CVI_SUCCESS) return s32Ret;

    // Measure quality time if needed
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    s32Ret = CVI_TDL_FaceQuality(args->stTDLHandle, stFrame, stFaceMeta, NULL);
    gettimeofday(&t1, NULL);
    // Optional: log execution time
    CHECK_ERROR(s32Ret, "CVI_TDL_FaceQuality");
    if (s32Ret != CVI_SUCCESS) return s32Ret;

    s32Ret = CVI_TDL_FaceRecognition(args->stTDLHandle, stFrame, stFaceMeta);
    CHECK_ERROR(s32Ret, "CVI_TDL_FaceRecognition");
    if (s32Ret != CVI_SUCCESS) return s32Ret;

    s32Ret = CVI_TDL_DeepSORT_Face(args->stTDLHandle, stFaceMeta, stTrackerMeta);
    CHECK_ERROR(s32Ret, "CVI_TDL_DeepSORT_Face");
    if (s32Ret != CVI_SUCCESS) return s32Ret;

    return CVI_SUCCESS;
}

static void *tdl_thread(void *args) {
    printf("[%s] INFO: TDL thread started\n", LOG_TAG);
    const tdl_thread_arg_t *tdl_args = (const tdl_thread_arg_t *)args;
    VIDEO_FRAME_INFO_S stFrame = {0};

    // Dynamic arrays
    uint64_t *processed_uids = NULL;
    size_t processed_uid_count = 0;
    size_t processed_uid_capacity = 0;

    tracked_id_t *tracked_ids = NULL;
    size_t tracked_id_count = 0;
    size_t tracked_id_capacity = TRACKED_ID_INIT_CAPACITY;

    const tracked_id_t *prev_tracked_ids = NULL;
    size_t prev_count = 0;

    while (!g_bExit) {
        CVI_S32 s32Ret = CVI_VPSS_GetChnFrame(0, 1, &stFrame, VPSS_FRAME_TIMEOUT_MS);
        if (s32Ret != CVI_SUCCESS) {
            CHECK_ERROR(s32Ret, "CVI_VPSS_GetChnFrame chn1");
            continue;
        }

        // THÊM: Lấy timestamp trước khi xử lý TDL
        struct timeval tv;
        gettimeofday(&tv, NULL);
        uint64_t frame_timestamp_us = (uint64_t)tv.tv_sec * 1000000 + tv.tv_usec;

        cvtdl_face_t stFaceMeta = {0};
        cvtdl_tracker_t stTrackerMeta = {0};

        s32Ret = detect_and_track(tdl_args, &stFrame, &stFaceMeta, &stTrackerMeta);
        if (s32Ret != CVI_SUCCESS) {
            CVI_TDL_Free(&stFaceMeta);
            CVI_TDL_Free(&stTrackerMeta);
            CVI_VPSS_ReleaseChnFrame(0, 1, &stFrame);
            continue;
        }

        // Log changes using previous tracked IDs
        log_tracker_changes(&stFaceMeta, &stTrackerMeta, prev_tracked_ids, prev_count);

        // Build new tracked IDs
        tracked_id_t *new_tracked_ids = malloc(stFaceMeta.size * sizeof(tracked_id_t));
        if (!new_tracked_ids && stFaceMeta.size > 0) {
            printf("[%s] ERROR: Failed to allocate new tracked IDs\n", LOG_TAG);
            CVI_TDL_Free(&stFaceMeta);
            CVI_TDL_Free(&stTrackerMeta);
            CVI_VPSS_ReleaseChnFrame(0, 1, &stFrame);
            continue;
        }
        size_t new_count = 0;

        for (uint32_t fid = 0; fid < stFaceMeta.size; fid++) {
            uint64_t curr_id = stFaceMeta.info[fid].unique_id;
            int curr_state = stTrackerMeta.info[fid].state;

            // Add to new tracked IDs
            if (new_count >= tracked_id_capacity) {
                tracked_id_capacity = tracked_id_capacity == 0 ? TRACKED_ID_INIT_CAPACITY : tracked_id_capacity * 2;
                tracked_id_t *temp = realloc(new_tracked_ids, tracked_id_capacity * sizeof(tracked_id_t));
                if (!temp) {
                    printf("[%s] ERROR: Failed to reallocate new tracked IDs\n", LOG_TAG);
                    free(new_tracked_ids);
                    CVI_TDL_Free(&stFaceMeta);
                    CVI_TDL_Free(&stTrackerMeta);
                    CVI_VPSS_ReleaseChnFrame(0, 1, &stFrame);
                    continue;
                }
                new_tracked_ids = temp;
            }
            new_tracked_ids[new_count].u_id = curr_id;
            new_tracked_ids[new_count].state = curr_state;
            new_count++;

            // Process cropping for STABLE faces with good quality
            if (curr_state == CVI_TRACKER_STABLE) {
                float frontal = fabs(stFaceMeta.info[fid].head_pose.roll) +
                                fabs(stFaceMeta.info[fid].head_pose.pitch) +
                                fabs(stFaceMeta.info[fid].head_pose.yaw);
                if (frontal >= 0 && frontal < HEAD_POSE_QUALITY_THRESHOLD) {
                    if (!is_uid_processed(curr_id, processed_uids, processed_uid_count)) {
                        // THAY ĐỔI: Truyền timestamp vào hàm crop_and_enqueue_face
                        s32Ret = crop_and_enqueue_face(&stFrame, &stFaceMeta.info[fid], &stTrackerMeta.info[fid], frame_timestamp_us);
                        if (s32Ret == CVI_SUCCESS) {
                            add_processed_uid(curr_id, &processed_uids, &processed_uid_count, &processed_uid_capacity);
                        }
                    }
                }
            }
        }

        // Log IDs that are no longer tracked
        for (size_t i = 0; i < prev_count; i++) {
            bool found = false;
            for (uint32_t fid = 0; fid < stFaceMeta.size; fid++) {
                if (prev_tracked_ids[i].u_id == stFaceMeta.info[fid].unique_id) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                printf("[%s] INFO: Face ID %" PRIu64 ": No longer tracked\n", LOG_TAG, prev_tracked_ids[i].u_id);
            }
        }

        // Update tracked IDs
        free(tracked_ids);
        tracked_ids = new_tracked_ids;
        tracked_id_count = new_count;
        tracked_id_capacity = new_count > 0 ? new_count : TRACKED_ID_INIT_CAPACITY;

        prev_tracked_ids = tracked_ids;
        prev_count = tracked_id_count;

        // Update global metadata
        pthread_mutex_lock(&g_result_mutex);
        CVI_TDL_CopyFaceMeta(&stFaceMeta, &g_stFaceMeta);
        CVI_TDL_CopyTrackerMeta(&stTrackerMeta, &g_stTrackerMeta);
        pthread_mutex_unlock(&g_result_mutex);

        CVI_TDL_Free(&stFaceMeta);
        CVI_TDL_Free(&stTrackerMeta);
        CVI_VPSS_ReleaseChnFrame(0, 1, &stFrame);
    }

    // Cleanup
    free(processed_uids);
    free(tracked_ids);
    printf("[%s] INFO: TDL thread exited\n", LOG_TAG);
    pthread_mutex_destroy(&g_result_mutex);
    pthread_exit(NULL);
}

static void log_tracker_changes(const cvtdl_face_t *stFaceMeta, const cvtdl_tracker_t *stTrackerMeta,
                                const tracked_id_t *prev_tracked_ids, size_t prev_count) {
    for (uint32_t i = 0; i < stFaceMeta->size; i++) {
        uint64_t curr_id = stFaceMeta->info[i].unique_id;
        int curr_state = stTrackerMeta->info[i].state;
        bool found = false;
        size_t prev_idx = 0;

        for (size_t j = 0; j < prev_count; j++) {
            if (prev_tracked_ids[j].u_id == curr_id) {
                found = true;
                prev_idx = j;
                break;
            }
        }

        if (!found || (found && prev_tracked_ids[prev_idx].state != curr_state)) {
            const char *state_str = curr_state == CVI_TRACKER_NEW ? "NEW" :
                                    curr_state == CVI_TRACKER_UNSTABLE ? "UNSTABLE" :
                                    curr_state == CVI_TRACKER_STABLE ? "STABLE" : "UNKNOWN";
            float frontal = fabs(stFaceMeta->info[i].head_pose.roll) +
                            fabs(stFaceMeta->info[i].head_pose.pitch) +
                            fabs(stFaceMeta->info[i].head_pose.yaw);
            printf("[%s] INFO: Face ID %" PRIu64 ": tracker_state = %s, frontal (roll + pitch + yaw) = %.2f\n",
                   LOG_TAG, curr_id, state_str, frontal);
        }
    }
}

static bool is_uid_processed(uint64_t uid, const uint64_t *processed_uids, size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (processed_uids[i] == uid) return true;
    }
    return false;
}

static void add_processed_uid(uint64_t uid, uint64_t **processed_uids, size_t *count, size_t *capacity) {
    if (*count >= *capacity) {
        *capacity = (*capacity == 0) ? PROCESSED_UID_INIT_CAPACITY : *capacity * 2;
        uint64_t *new_uids = realloc(*processed_uids, *capacity * sizeof(uint64_t));
        if (!new_uids) {
            printf("[%s] ERROR: Failed to reallocate processed UIDs\n", LOG_TAG);
            return;
        }
        *processed_uids = new_uids;
    }
    (*processed_uids)[*count] = uid;
    (*count)++;
}


static void update_visit_txt(uint64_t person_id, uint32_t visit_count) {
    char txt_filename[MAX_FILENAME_LEN];
    snprintf(txt_filename, MAX_FILENAME_LEN, "/mnt/data/faces/person_%" PRIu64 ".txt", person_id);

    // Mở file để đọc nội dung gốc
    FILE *txt_file = fopen(txt_filename, "r");
    if (!txt_file) {
        printf("[%s] ERROR: Failed to open %s for reading\n", LOG_TAG, txt_filename);
        return;
    }

    char line1[512]; // Buffer cho dòng 1
    char landmarks[5][128]; // Buffer cho 5 dòng landmarks
    if (fgets(line1, sizeof(line1), txt_file) == NULL) {
        printf("[%s] ERROR: Failed to read line 1 from %s\n", LOG_TAG, txt_filename);
        fclose(txt_file);
        return;
    }

    for (int i = 0; i < 5; i++) {
        if (fgets(landmarks[i], sizeof(landmarks[i]), txt_file) == NULL) {
            printf("[%s] ERROR: Failed to read landmark line %d from %s\n", LOG_TAG, i+1, txt_filename);
            fclose(txt_file);
            return;
        }
    }
    fclose(txt_file);

    // Parse dòng 1 để lấy các fields trừ VisitCount
    uint64_t parsed_person_id, parsed_uid;
    uint32_t parsed_counter;
    float parsed_x1, parsed_y1, parsed_width, parsed_height, parsed_quality;
    uint32_t old_visit_count; // Để parse nhưng không dùng
    int parsed = sscanf(line1, "PersonID:%" SCNu64 ", TrackerUID:%" SCNu64 ", Count:%" SCNu32 ", BBox:%f,%f,%f,%f, Quality:%f, VisitCount:%" SCNu32 "\n",
                        &parsed_person_id, &parsed_uid, &parsed_counter,
                        &parsed_x1, &parsed_y1, &parsed_width, &parsed_height,
                        &parsed_quality, &old_visit_count);

    if (parsed < 9) { // Ít nhất 9 fields trước VisitCount
        printf("[%s] ERROR: Failed to parse original line 1 in %s\n", LOG_TAG, txt_filename);
        return;
    }

    // Mở lại để viết (overwrite)
    txt_file = fopen(txt_filename, "w");
    if (!txt_file) {
        printf("[%s] ERROR: Failed to open %s for updating\n", LOG_TAG, txt_filename);
        return;
    }

    // Viết lại dòng 1 với VisitCount mới
    fprintf(txt_file, "PersonID:%" PRIu64 ", TrackerUID:%" PRIu64 ", Count:%u, BBox:%.2f,%.2f,%.2f,%.2f, Quality:%.2f, VisitCount:%u\n",
            parsed_person_id, parsed_uid, parsed_counter,
            parsed_x1, parsed_y1, parsed_width, parsed_height,
            parsed_quality, visit_count);

    // Viết lại các dòng landmarks
    for (int i = 0; i < 5; i++) {
        fprintf(txt_file, "%s", landmarks[i]);
    }

    fflush(txt_file);
    fclose(txt_file);
    printf("[%s] INFO: Updated %s with VisitCount=%u\n", LOG_TAG, txt_filename, visit_count);
}

static CVI_S32 crop_and_enqueue_face(VIDEO_FRAME_INFO_S *stFrame,
                                     const cvtdl_face_info_t *face_info,
                                     const cvtdl_tracker_info_t *tracker_info,
                                     uint64_t timestamp_us) {
    cvtdl_bbox_t bbox = face_info->bbox;

    // ====================== CẢI TIẾN: PADDING + KIỂM TRA KÍCH THƯỚC ======================
    const float EXTEN_RATIO = 1.45f;      // Padding 1.45× (tốt nhất: 1.3 ~ 1.6)
    const uint32_t MIN_FACE_SIZE = 96;    // Bỏ qua nếu sau padding vẫn quá nhỏ

    float w = bbox.x2 - bbox.x1;
    float h = bbox.y2 - bbox.y1;
    float cx = (bbox.x1 + bbox.x2) / 2.0f;
    float cy = (bbox.y1 + bbox.y2) / 2.0f;

    // Tạo bbox mới có padding
    bbox.x1 = fmaxf(0.0f, cx - w * EXTEN_RATIO / 2.0f);
    bbox.y1 = fmaxf(0.0f, cy - h * EXTEN_RATIO / 2.0f);
    bbox.x2 = fminf((float)stFrame->stVFrame.u32Width,  cx + w * EXTEN_RATIO / 2.0f);
    bbox.y2 = fminf((float)stFrame->stVFrame.u32Height, cy + h * EXTEN_RATIO / 2.0f);

    float crop_w = bbox.x2 - bbox.x1;
    float crop_h = bbox.y2 - bbox.y1;

    if (crop_w < MIN_FACE_SIZE || crop_h < MIN_FACE_SIZE) {
        printf("[%s] WARNING: Face too small after padding (%.0fx%.0f px) → skip\n",
               LOG_TAG, crop_w, crop_h);
        return CVI_SUCCESS;
    }

    printf("[%s] INFO: Crop with padding %.2fx → size %.0fx%.0f\n",
           LOG_TAG, EXTEN_RATIO, crop_w, crop_h);

    // ====================== CROP VỚI PADDING (hàm thư viện chính thức) ======================
    cvtdl_image_t cropped_image = {0};
    float offset_x = 0.0f, offset_y = 0.0f;   // không dùng offset thì truyền NULL cũng được

    CVI_S32 s32Ret = CVI_TDL_CropImage_Exten(stFrame, &cropped_image, &bbox,
                                              true,          // cvtRGB888 = true
                                              EXTEN_RATIO,   // padding ratio
                                              &offset_x, &offset_y);
    CHECK_ERROR(s32Ret, "CVI_TDL_CropImage_Exten");
    if (s32Ret != CVI_SUCCESS) return s32Ret;

    // ====================== PHẦN LOGIC CŨ (DB + enqueue) GIỮ NGUYÊN ======================
    io_data_t data = {
        .u_id = face_info->unique_id,
        .quality = face_info->face_quality,
        .counter = g_face_counter++,
        .bbox = bbox,                    // bbox sau padding (dùng ghi metadata)
        .pts = face_info->pts,
        .feature = {0},
        .timestamp_us = timestamp_us,
        .person_id = 0
    };

    CVI_TDL_CopyImage(&cropped_image, &data.image);
    CVI_TDL_Free(&cropped_image);        // Giải phóng ngay sau khi copy

    // Copy feature vector
    data.feature.size = face_info->feature.size;
    data.feature.type = face_info->feature.type;
    if (data.feature.size > 0) {
        size_t elem_size = (data.feature.type == TYPE_INT8) ? sizeof(int8_t) : sizeof(float);
        data.feature.ptr = malloc(elem_size * data.feature.size);
        if (!data.feature.ptr) {
            printf("[%s] ERROR: Failed to allocate feature\n", LOG_TAG);
            CVI_TDL_Free(&data.image);
            return CVI_FAILURE;
        }
        memcpy(data.feature.ptr, face_info->feature.ptr, elem_size * data.feature.size);
    }

    // ====================== SO SÁNH FEATURE VỚI DB ======================
    pthread_mutex_lock(&g_db_mutex);
    float max_sim = 0.0f;
    int matched_idx = -1;
    for (size_t i = 0; i < g_db_count; i++) {
        float sim = cosine_similarity(&data.feature, &g_face_db[i].feature);
        if (sim > max_sim) {
            max_sim = sim;
            matched_idx = (int)i;
        }
    }

    if (max_sim > COSINE_THRESHOLD && matched_idx != -1) {
        // === NGƯỜI CŨ ===
        g_face_db[matched_idx].visit_count++;
        g_face_db[matched_idx].last_seen_ts = timestamp_us;
        printf("[%s] RE-VISIT DETECTED! Guest_ID %" PRIu64 " visited %u times\n",
               LOG_TAG, g_face_db[matched_idx].person_id, g_face_db[matched_idx].visit_count);
        update_visit_txt(g_face_db[matched_idx].person_id, g_face_db[matched_idx].visit_count);
        g_total_visits_all++;

        CVI_TDL_Free(&data.image);
        if (data.feature.ptr) free(data.feature.ptr);
        pthread_mutex_unlock(&g_db_mutex);
        return CVI_SUCCESS;
    } else {
        // === NGƯỜI MỚI ===
        if (g_db_count >= MAX_DB_SIZE) {
            printf("[%s] WARNING: DB full\n", LOG_TAG);
            pthread_mutex_unlock(&g_db_mutex);
            CVI_TDL_Free(&data.image);
            if (data.feature.ptr) free(data.feature.ptr);
            return CVI_FAILURE;
        }

        uint64_t new_person_id = g_next_person_id++;
        g_total_people++;
        g_face_db[g_db_count].person_id = new_person_id;
        g_face_db[g_db_count].visit_count = 1;
        g_face_db[g_db_count].last_seen_ts = timestamp_us;
        strcpy(g_face_db[g_db_count].name, "Unknown");

        g_face_db[g_db_count].feature.size = data.feature.size;
        g_face_db[g_db_count].feature.type = data.feature.type;
        g_face_db[g_db_count].feature.ptr = malloc(data.feature.size * sizeof(int8_t));
        memcpy(g_face_db[g_db_count].feature.ptr, data.feature.ptr, data.feature.size * sizeof(int8_t));

        g_db_count++;
        g_total_visits_all++;

        printf("[%s] NEW PERSON! Person ID=%" PRIu64 " (UID=%" PRIu64 ")\n",
               LOG_TAG, new_person_id, data.u_id);

        data.person_id = new_person_id;
        enqueue_io_data(&data);
        pthread_mutex_unlock(&g_db_mutex);
        return CVI_SUCCESS;
    }
}

static void signal_handler(CVI_S32 signo) {
    printf("[%s] INFO: Received signal %d\n", LOG_TAG, signo);
    if (signo == SIGINT || signo == SIGTERM) {
        g_bExit = true;
        g_bRunImageWriter = false;

        pthread_mutex_lock(&g_db_mutex);
        for (size_t i = 0; i < g_db_count; i++) {
            update_visit_txt(g_face_db[i].person_id, g_face_db[i].visit_count);
        }
        pthread_mutex_unlock(&g_db_mutex);
    }
}

static CVI_S32 init_system(const char *retina_model_path, const char *quality_model_path, const char *recognition_model_path,
                           PIXEL_FORMAT_E enInputFormat, SAMPLE_TDL_MW_CONFIG_S *stMWConfig) {
    CVI_S32 s32Ret = SAMPLE_TDL_Get_VI_Config(&stMWConfig->stViConfig);
    CHECK_ERROR(s32Ret, "SAMPLE_TDL_Get_VI_Config");
    if (s32Ret != CVI_SUCCESS || stMWConfig->stViConfig.s32WorkingViNum <= 0) {
        printf("[%s] ERROR: Failed to get VI config from /mnt/data/sensor_cfg.ini\n", LOG_TAG);
        return -1;
    }

    PIC_SIZE_E enPicSize;
    s32Ret = SAMPLE_COMM_VI_GetSizeBySensor(stMWConfig->stViConfig.astViInfo[0].stSnsInfo.enSnsType, &enPicSize);
    CHECK_ERROR(s32Ret, "SAMPLE_COMM_VI_GetSizeBySensor");

    SIZE_S stSensorSize;
    s32Ret = SAMPLE_COMM_SYS_GetPicSize(enPicSize, &stSensorSize);
    CHECK_ERROR(s32Ret, "SAMPLE_COMM_SYS_GetPicSize");

    SIZE_S stVencSize = { .u32Width = 1920, .u32Height = 1080 };

    // VB Pool setup
    stMWConfig->stVBPoolConfig.u32VBPoolCount = 3;
    // Pool 0: VI bind to VPSS_CHN0
    stMWConfig->stVBPoolConfig.astVBPoolSetup[0].enFormat = VI_PIXEL_FORMAT;
    stMWConfig->stVBPoolConfig.astVBPoolSetup[0].u32BlkCount = 3;
    stMWConfig->stVBPoolConfig.astVBPoolSetup[0].u32Height = stSensorSize.u32Height;
    stMWConfig->stVBPoolConfig.astVBPoolSetup[0].u32Width = stSensorSize.u32Width;
    stMWConfig->stVBPoolConfig.astVBPoolSetup[0].bBind = true;
    stMWConfig->stVBPoolConfig.astVBPoolSetup[0].u32VpssChnBinding = VPSS_CHN0;
    stMWConfig->stVBPoolConfig.astVBPoolSetup[0].u32VpssGrpBinding = (VPSS_GRP)0;
    // Pool 1: Input format bind to VPSS_CHN1
    stMWConfig->stVBPoolConfig.astVBPoolSetup[1].enFormat = enInputFormat;
    stMWConfig->stVBPoolConfig.astVBPoolSetup[1].u32BlkCount = 3;
    stMWConfig->stVBPoolConfig.astVBPoolSetup[1].u32Height = stVencSize.u32Height;
    stMWConfig->stVBPoolConfig.astVBPoolSetup[1].u32Width = stVencSize.u32Width;
    stMWConfig->stVBPoolConfig.astVBPoolSetup[1].bBind = true;
    stMWConfig->stVBPoolConfig.astVBPoolSetup[1].u32VpssChnBinding = VPSS_CHN1;
    stMWConfig->stVBPoolConfig.astVBPoolSetup[1].u32VpssGrpBinding = (VPSS_GRP)0;
    // Pool 2: RGB planar unbound
    stMWConfig->stVBPoolConfig.astVBPoolSetup[2].enFormat = PIXEL_FORMAT_RGB_888_PLANAR;
    stMWConfig->stVBPoolConfig.astVBPoolSetup[2].u32BlkCount = 1;
    stMWConfig->stVBPoolConfig.astVBPoolSetup[2].u32Height = 720;
    stMWConfig->stVBPoolConfig.astVBPoolSetup[2].u32Width = 1280;
    stMWConfig->stVBPoolConfig.astVBPoolSetup[2].bBind = false;

    // VPSS setup
    stMWConfig->stVPSSPoolConfig.u32VpssGrpCount = 1;
#ifndef CV186X
    stMWConfig->stVPSSPoolConfig.stVpssMode.aenInput[0] = VPSS_INPUT_MEM;
    stMWConfig->stVPSSPoolConfig.stVpssMode.enMode = VPSS_MODE_DUAL;
    stMWConfig->stVPSSPoolConfig.stVpssMode.ViPipe[0] = 0;
    stMWConfig->stVPSSPoolConfig.stVpssMode.aenInput[1] = VPSS_INPUT_ISP;
    stMWConfig->stVPSSPoolConfig.stVpssMode.ViPipe[1] = 0;
#endif

    SAMPLE_TDL_VPSS_CONFIG_S *vpss_config = &stMWConfig->stVPSSPoolConfig.astVpssConfig[0];
    vpss_config->bBindVI = true;
    VPSS_GRP_DEFAULT_HELPER2(&vpss_config->stVpssGrpAttr, stSensorSize.u32Width, stSensorSize.u32Height, VI_PIXEL_FORMAT, 1);
    vpss_config->u32ChnCount = 2;
    vpss_config->u32ChnBindVI = 0;
    VPSS_CHN_DEFAULT_HELPER(&vpss_config->astVpssChnAttr[0], stVencSize.u32Width, stVencSize.u32Height, VI_PIXEL_FORMAT, true);
    VPSS_CHN_DEFAULT_HELPER(&vpss_config->astVpssChnAttr[1], stVencSize.u32Width, stVencSize.u32Height, enInputFormat, true);
    vpss_config->astVpssChnAttr[0].bMirror = CVI_TRUE; // Bật Mirror (lật ngang)
    vpss_config->astVpssChnAttr[0].bFlip = CVI_TRUE;   // Bật Flip (lật dọc)
    vpss_config->astVpssChnAttr[1].bMirror = CVI_TRUE; // Bật Mirror (lật ngang)
    vpss_config->astVpssChnAttr[1].bFlip = CVI_TRUE;   // Bật Flip (lật dọc)

    // VENC and RTSP config
    SAMPLE_TDL_Get_Input_Config(&stMWConfig->stVencConfig.stChnInputCfg);
    stMWConfig->stVencConfig.u32FrameWidth = stVencSize.u32Width;
    stMWConfig->stVencConfig.u32FrameHeight = stVencSize.u32Height;
    SAMPLE_TDL_Get_RTSP_Config(&stMWConfig->stRTSPConfig.stRTSPConfig);

    return CVI_SUCCESS;
}

static CVI_S32 init_tdl(cvitdl_handle_t *stTDLHandle, cvitdl_service_handle_t *stServiceHandle, const char *retina_model_path, const char *quality_model_path, const char *recognition_model_path) {
    CVI_S32 s32Ret = CVI_TDL_CreateHandle2(stTDLHandle, 1, 0);
    CHECK_ERROR(s32Ret, "CVI_TDL_CreateHandle2");

    s32Ret = CVI_TDL_SetVBPool(*stTDLHandle, 0, 2);
    CHECK_ERROR(s32Ret, "CVI_TDL_SetVBPool");
    CVI_TDL_SetVpssTimeout(*stTDLHandle, TDL_VPSS_TIMEOUT_MS);

    s32Ret = CVI_TDL_Service_CreateHandle(stServiceHandle, *stTDLHandle);
    CHECK_ERROR(s32Ret, "CVI_TDL_Service_CreateHandle");

    s32Ret = CVI_TDL_OpenModel(*stTDLHandle, CVI_TDL_SUPPORTED_MODEL_SCRFDFACE, retina_model_path);
    CHECK_ERROR(s32Ret, "CVI_TDL_OpenModel (RetinaFace)");

    s32Ret = CVI_TDL_OpenModel(*stTDLHandle, CVI_TDL_SUPPORTED_MODEL_FACEQUALITY, quality_model_path);
    CHECK_ERROR(s32Ret, "CVI_TDL_OpenModel (FaceQuality)");

    s32Ret = CVI_TDL_OpenModel(*stTDLHandle, CVI_TDL_SUPPORTED_MODEL_FACERECOGNITION, recognition_model_path);
    CHECK_ERROR(s32Ret, "CVI_TDL_OpenModel (FaceRecognition)");

    CVI_TDL_DeepSORT_Init(*stTDLHandle, true);
    cvtdl_deepsort_config_t ds_conf;
    CVI_TDL_DeepSORT_GetDefaultConfig(&ds_conf);
    set_sample_mot_config(&ds_conf);
    s32Ret = CVI_TDL_DeepSORT_SetConfig(*stTDLHandle, &ds_conf, -1, false);
    CHECK_ERROR(s32Ret, "CVI_TDL_DeepSORT_SetConfig");

    return CVI_SUCCESS;
}

static void cleanup_system(SAMPLE_TDL_MW_CONTEXT *stMWContext, cvitdl_handle_t stTDLHandle, cvitdl_service_handle_t stServiceHandle) {
    CVI_TDL_Service_DestroyHandle(stServiceHandle);
    CVI_TDL_DestroyHandle(stTDLHandle);
    SAMPLE_TDL_Destroy_MW(stMWContext);
    CVI_VPSS_DestroyGrp(0);
    CVI_VI_DestroyPipe(0);
    CVI_SYS_Exit();
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("Usage: %s <RETINA_MODEL_PATH> <QUALITY_MODEL_PATH> <RECOGNITION_MODEL_PATH> <INPUT_FORMAT>\n"
        "\tINPUT_FORMAT: 0=RGB888, 1=NV21, 2=YUV420\n", argv[0]);
        return CVI_TDL_FAILURE;
    }
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    PIXEL_FORMAT_E enInputFormat;
    int format = atoi(argv[4]);
    if (format == 0) {
        enInputFormat = PIXEL_FORMAT_RGB_888;
    } else if (format == 1) {
        enInputFormat = PIXEL_FORMAT_NV21;
    } else if (format == 2) {
        enInputFormat = PIXEL_FORMAT_YUV_PLANAR_420;
    } else {
        printf("[%s] ERROR: Unknown input format %d\n", LOG_TAG, format);
        return CVI_FAILURE;
    }
    SAMPLE_TDL_MW_CONFIG_S stMWConfig = {0};
    CVI_S32 s32Ret = init_system(argv[1], argv[2], argv[3], enInputFormat, &stMWConfig);
    if (s32Ret != CVI_SUCCESS) return -1;
    SAMPLE_TDL_MW_CONTEXT stMWContext = {0};
    // THAY ĐỔI: Sửa typo từ SAMPLE_TDL_Init_MW thành SAMPLE_TDL_Init_WM
    s32Ret = SAMPLE_TDL_Init_WM(&stMWConfig, &stMWContext);
    CHECK_ERROR(s32Ret, "SAMPLE_TDL_Init_WM");
    if (s32Ret != CVI_SUCCESS) return -1;
    cvitdl_handle_t stTDLHandle = NULL;
    cvitdl_service_handle_t stServiceHandle = NULL;
    s32Ret = init_tdl(&stTDLHandle, &stServiceHandle, argv[1], argv[2], argv[3]);
    if (s32Ret != CVI_SUCCESS) goto cleanup;
    s32Ret = load_face_db();
    if (s32Ret != CVI_SUCCESS) goto cleanup;
    // Threads
    pthread_t venc_tid, tdl_tid, io_tid, db_tid;
    venc_thread_arg_t venc_args = { .pstMWContext = &stMWContext, .stServiceHandle = stServiceHandle };
    tdl_thread_arg_t tdl_args = { .stTDLHandle = stTDLHandle, .bTrackingWithFeature = false };
    pthread_create(&venc_tid, NULL, venc_thread, &venc_args);
    pthread_create(&tdl_tid, NULL, tdl_thread, &tdl_args);
    pthread_create(&io_tid, NULL, image_writer_thread, NULL);
    // Thêm thread reload DB
    pthread_create(&db_tid, NULL, db_reload_thread, NULL);
    pthread_join(venc_tid, NULL);
    pthread_join(tdl_tid, NULL);
    pthread_join(io_tid, NULL);
    pthread_join(db_tid, NULL);
cleanup:
    printf("[%s] INFO: Total_guests: %u, Total_visit: %u\n", LOG_TAG, g_total_people, g_total_visits_all);
    for (size_t i = 0; i < g_db_count; i++) {
        printf("[%s] Guest %" PRIu64 "visited %u times \n",
            LOG_TAG, g_face_db[i].person_id, g_face_db[i].visit_count);
    }
    // Thêm cleanup DB
    for (size_t i = 0; i < g_db_count; i++) {
        if (g_face_db[i].feature.ptr) free(g_face_db[i].feature.ptr);
    }
    pthread_mutex_destroy(&g_db_mutex);
    cleanup_system(&stMWContext, stTDLHandle, stServiceHandle);
    printf("[%s] INFO: Exited with status %#x. Total faces: %u\n", LOG_TAG, s32Ret, g_face_counter);
    return s32Ret;
}
