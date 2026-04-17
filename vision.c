#include <stdio.h>    // fopen, fclose, fread, fwrite, fprintf, printf, fgets, sscanf 
#include <stdlib.h>   // malloc, calloc, free, exit, atof 
#include <string.h>   // strcmp, strncmp, memcpy 
#include <math.h>     // sqrtf, atan2f, M_PI 
#include <stdint.h>   // uint8_t — гарантований 8-бітний беззнаковий тип 
#ifdef _WIN32
#include <fcntl.h>    // _O_BINARY 
#include <io.h>       // _setmode, _fileno 
#endif

typedef struct {
    int width;       // ширина зображення в пікселях 
    int height;      // висота зображення в пікселях 
    uint8_t *data;   // масив яскравостей, розмір = width * height байт 
} Image;

static inline uint8_t clamp_u8(float v) {
    if (v < 0.0f)   return 0;    // від'ємні → чорний 
    if (v > 255.0f) return 255;  // більше 255 → білий 
    return (uint8_t)v;           // відрізаємо дробову частину 
}


static inline int idx(const Image *img, int x, int y) { //перетворює 2D-координати (x, y) в індекс плаского масиву
    return y * img->width + x;
}

static inline uint8_t get_pixel(const Image *img, int x, int y) { //читає піксель з координатами (x, y)
    if (x < 0)            x = -x;                        // ліва межа: відзеркалюємо 
    if (y < 0)            y = -y;                        // верхня межа: відзеркалюємо 
    if (x >= img->width)  x = 2 * img->width  - x - 2;  // права межа: відзеркалюємо 
    if (y >= img->height) y = 2 * img->height - y - 2;  // нижня межа: відзеркалюємо 
    return img->data[idx(img, x, y)];                    // повертаємо піксель за індексом 
}

static Image *alloc_image(int w, int h) { //виділяє нову Image заданого розміру, заповнену нулями
    Image *img = malloc(sizeof(Image));                    // виділяємо саму структуру 
    if (!img) { fprintf(stderr, "OOM\n"); exit(1); }      // OOM = Out Of Memory 
    img->width  = w;                                       // зберігаємо ширину 
    img->height = h;                                       // зберігаємо висоту 
    img->data   = calloc((size_t)w * h, 1);                // w*h байт, всі = 0 
    if (!img->data) { fprintf(stderr, "OOM\n"); exit(1); }
    return img;
}

static void free_image(Image *img) { //звільняє пам'ять Image
    if (img) {           // захист від free(NULL) — NULL перевіряємо вручну 
        free(img->data); // звільняємо масив пікселів 
        free(img);       // звільняємо саму структуру 
    }
}

static void ppm_readline(FILE *f, char *buf, int size) { //читає рядок із файлу, пропускаючи рядки-коментарі 
    do {
        if (!fgets(buf, size, f)) { // читаємо рядок у буфер 
            buf[0] = '\0';          // якщо EOF або помилка — повертаємо порожній рядок 
            return;
        }
    } while (buf[0] == '#');        // повторюємо поки рядок починається з '#' 
}

Image *read_image(const char *path) { //читає P6 (RGB) або P5 (grayscale) PPM/PGM файл
    FILE *f = fopen(path, "rb");  // "rb" = read binary, важливо на Windows 
    if (!f) {
        fprintf(stderr, "Не можу відкрити '%s'\n", path);
        return NULL;
    }

    char magic[8];                              // буфер для magic number 
    ppm_readline(f, magic, sizeof(magic));      // читаємо перший рядок заголовка 

    int is_rgb = 0;                             // прапорець: 1=RGB, 0=grayscale 
    if      (strncmp(magic, "P6", 2) == 0) is_rgb = 1;   // P6 = кольоровий PPM 
    else if (strncmp(magic, "P5", 2) == 0) is_rgb = 0;   // P5 = сірий PGM 
    else {
        fprintf(stderr, "Підтримується тільки P5/P6 PPM/PGM\n");
        fclose(f);
        return NULL;
    }

    char buf[64];
    ppm_readline(f, buf, sizeof(buf));          // читаємо рядок з розмірами 

    int w, h;
    if (sscanf(buf, "%d %d", &w, &h) != 2) {
        // Деякі генератори пишуть ширину і висоту на окремих рядках
        int w2;
        if (sscanf(buf, "%d", &w2) != 1) {
            fprintf(stderr, "Поганий заголовок\n");
            fclose(f);
            return NULL;
        }
        w = w2;
        ppm_readline(f, buf, sizeof(buf));      // читаємо висоту окремо 
        sscanf(buf, "%d", &h);
    }

    ppm_readline(f, buf, sizeof(buf));          // читаємо рядок maxval (255) — ігноруємо
    Image *img = alloc_image(w, h);             // виділяємо пам'ять під зображення

    if (is_rgb) {
        uint8_t rgb[3];                         // тимчасовий буфер для одного RGB-пікселя
        for (int i = 0; i < w * h; i++) {       // проходимо по всіх пікселях
            if (fread(rgb, 1, 3, f) != 3) break;  // читаємо 3 байти: R, G, B
            img->data[i] = clamp_u8(0.299f * rgb[0] + 0.587f * rgb[1] + 0.114f * rgb[2]);
        }
    } else {
        fread(img->data, 1, (size_t)w * h, f);
    }

    fclose(f);
    return img;
}

static Image *read_image_fp(FILE *f) {
    char magic[8];
    ppm_readline(f, magic, sizeof(magic));
    if (magic[0] == '\0') return NULL;

    int is_rgb = 0;
    if      (strncmp(magic, "P6", 2) == 0) is_rgb = 1;
    else if (strncmp(magic, "P5", 2) == 0) is_rgb = 0;
    else { fprintf(stderr, "Підтримується тільки P5/P6\n"); return NULL; }

    char buf[64];
    ppm_readline(f, buf, sizeof(buf));
    int w, h;
    if (sscanf(buf, "%d %d", &w, &h) != 2) {
        int w2;
        if (sscanf(buf, "%d", &w2) != 1) { fprintf(stderr, "Поганий заголовок\n"); return NULL; }
        w = w2;
        ppm_readline(f, buf, sizeof(buf));
        sscanf(buf, "%d", &h);
    }
    ppm_readline(f, buf, sizeof(buf)); 

    Image *img = alloc_image(w, h);
    if (is_rgb) {
        uint8_t rgb[3];
        for (int i = 0; i < w * h; i++) {
            if (fread(rgb, 1, 3, f) != 3) break;
            img->data[i] = clamp_u8(0.299f * rgb[0] + 0.587f * rgb[1] + 0.114f * rgb[2]);
        }
    } else {
        fread(img->data, 1, (size_t)w * h, f);
    }
    return img;
}

int write_image(const char *path, const Image *img) { //записує Image у файл
    FILE *f = fopen(path, "wb");       // "wb" = write binary 
    if (!f) {
        fprintf(stderr, "Не можу записати '%s'\n", path);
        return 0;
    }
    fprintf(f, "P5\n%d %d\n255\n",    // пишемо текстовий заголовок PGM 
            img->width, img->height);
    fwrite(img->data, 1,               // пишемо всі байти пікселів одним разом 
           (size_t)img->width * img->height, f);
    fclose(f);
    return 1;
}

static int write_image_fp(FILE *f, const Image *img) {
    fprintf(f, "P5\n%d %d\n255\n", img->width, img->height);
    fwrite(img->data, 1, (size_t)img->width * img->height, f);
    fflush(f);
    return 1;
}

static Image *make_sidebyside(const Image *left, const Image *right) { //створює нове зображення, яке містить два зображення поруч (side-by-side)
    int h = left->height < right->height ? left->height : right->height;
    int w = left->width + right->width;
    Image *out = alloc_image(w, h);
    for (int y = 0; y < h; y++) {
        memcpy(out->data + y * w,               left->data  + y * left->width,  left->width);
        memcpy(out->data + y * w + left->width, right->data + y * right->width, right->width);
    }
    return out;
}

static const float GAUSS5[5][5] = {
    { 2,  4,  5,  4,  2},   // верхній рядок (найдальше від центру — найменший вплив) 
    { 4,  9, 12,  9,  4},
    { 5, 12, 15, 12,  5},   // центральний рядок (найбільші коефіцієнти) 
    { 4,  9, 12,  9,  4},
    { 2,  4,  5,  4,  2},   // нижній рядок 
};
static const float GAUSS5_SUM = 159.0f; // сума всіх елементів ядра, для нормалізації

static Image *gaussian_blur(const Image *src) { // повертає нове розмите зображення
    Image *dst = alloc_image(src->width, src->height); // результат того ж розміру 

    for (int y = 0; y < src->height; y++) {       // рядок за рядком 
        for (int x = 0; x < src->width; x++) {    // піксель за пікселем 
            float acc = 0.0f;                       // акумулятор зваженої суми 

            for (int ky = 0; ky < 5; ky++)          // ky = рядок ядра (0..4) 
                for (int kx = 0; kx < 5; kx++)      // kx = стовпець ядра (0..4) 
                    acc += GAUSS5[ky][kx] * get_pixel(src, x + kx - 2, y + ky - 2);
            dst->data[idx(dst, x, y)] = clamp_u8(acc / GAUSS5_SUM);
        }
    }
    return dst;
}

Image *sobel(const Image *src) { //повертає карту сили контурів
    static const int Kx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    static const int Ky[3][3] = { {-1,-2,-1}, { 0, 0, 0}, { 1, 2, 1} };

    Image *dst = alloc_image(src->width, src->height); // результат того ж розміру 

    for (int y = 0; y < src->height; y++) {       // проходимо по рядках 
        for (int x = 0; x < src->width; x++) {    // проходимо по стовпцях 
            float gx = 0.0f;
            float gy = 0.0f;  

            for (int ky = 0; ky < 3; ky++)          // рядок ядра 3x3 
                for (int kx = 0; kx < 3; kx++) {    // стовпець ядра 3x3 
                    float p = get_pixel(src, x + kx - 1, y + ky - 1); // сусідній піксель 
                    gx += Kx[ky][kx] * p;  // додаємо внесок у горизонтальний градієнт 
                    gy += Ky[ky][kx] * p;  // додаємо внесок у вертикальний градієнт 
                }
            dst->data[idx(dst, x, y)] = clamp_u8(sqrtf(gx*gx + gy*gy));
        }
    }
    return dst;
}

static inline int quantize_angle(float angle_rad) { //округлює кут градієнта
    float deg = angle_rad * (180.0f / (float)M_PI); // радіани - градуси 
    if (deg < 0) deg += 180.0f;
    if      (deg <  22.5f || deg >= 157.5f) return 0;    // горизонталь
    else if (deg <  67.5f)                  return 45;   // діагональ
    else if (deg < 112.5f)                  return 90;   // вертикаль
    else                                    return 135;  // діагональ
}

Image *canny(const Image *src, float low_t, float high_t) { //повний пайплайн алгоритму Канні
    int w = src->width;    // ширина 
    int h = src->height;   // висота 
    int n = w * h;         // загальна кількість пікселів 

    Image *blurred = gaussian_blur(src);

    // ті самі ядра Собеля що і у функції sobel
    static const int Kx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    static const int Ky[3][3] = { {-1,-2,-1}, { 0, 0, 0}, { 1, 2, 1} };

    float *magnitude = malloc(n * sizeof(float));  // сила градієнта для кожного пікселя
    int   *direction = malloc(n * sizeof(int));    // квантований напрямок (0/45/90/135)
    if (!magnitude || !direction) { fprintf(stderr, "OOM\n"); exit(1); }

    float max_mag = 0.0f;  // максимальна сила градієнта на всій картинці
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float gx = 0.0f, gy = 0.0f;   // горизонтальний і вертикальний градієнти

            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++) {
                    float p = get_pixel(blurred, x + kx - 1, y + ky - 1); // беремо з розмитого
                    gx += Kx[ky][kx] * p;  // внесок у горизонтальний градієнт
                    gy += Ky[ky][kx] * p;  // внесок у вертикальний градієнт
                }

            float mag = sqrtf(gx*gx + gy*gy);    // сила контуру в цій точці
            magnitude[y * w + x] = mag;            
            direction[y * w + x] = quantize_angle(atan2f(gy, gx)); // зберігаємо напрямок
            if (mag > max_mag) max_mag = mag;      // оновлюємо максимум для нормалізації порогів 
        }
    }

    free_image(blurred);  // розмите зображення більше не потрібне

    // перетворюємо відносні пороги (0.0-1.0) в абсолютні значення градієнта
    float low  = low_t  * max_mag;
    float high = high_t * max_mag;

    float *nms = calloc(n, sizeof(float)); 
    if (!nms) { fprintf(stderr, "OOM\n"); exit(1); }

    
    for (int y = 1; y < h - 1; y++) {// пропускаємо крайові пікселі (y=0, y=h-1, x=0, x=w-1)
        for (int x = 1; x < w - 1; x++) {
            float m = magnitude[y * w + x];  // сила поточного пікселя 
            float a, b; 

            // вибираємо пару сусідів залежно від напрямку контуру:
            //    0  - сусіди зліва/справа      (горизонтальний контур)
            //   45  - сусіди по діагоналі
            //   90  - сусіди зверху/знизу       (вертикальний контур)
            //  135  - сусіди по іншій діагоналі
            switch (direction[y * w + x]) {
                case   0: a = magnitude[y * w + x - 1];       // ліво 
                          b = magnitude[y * w + x + 1];       // право 
                          break;
                case  45: a = magnitude[(y-1)*w + x+1];       // вверх-право 
                          b = magnitude[(y+1)*w + x-1];       // вниз-ліво 
                          break;
                case  90: a = magnitude[(y-1)*w + x];         // вверх 
                          b = magnitude[(y+1)*w + x];         // вниз 
                          break;
                default:  a = magnitude[(y-1)*w + x-1];       // вверх-ліво 
                          b = magnitude[(y+1)*w + x+1];       // вниз-право 
                          break;
            }

            nms[y * w + x] = (m >= a && m >= b) ? m : 0.0f;
        }
    }

    free(magnitude);
    free(direction);

    uint8_t *edge = calloc(n, 1);  // масив міток, спочатку всі = 0 
    if (!edge) { fprintf(stderr, "OOM\n"); exit(1); }

    for (int i = 0; i < n; i++) {
        if      (nms[i] >= high) edge[i] = 255;  // сильний - точно контур
        else if (nms[i] >= low)  edge[i] = 128;  // кандидат - може бути контуром
    }

    free(nms);  // nms більше не потрібна

    int changed = 1;          // прапорець: чи були зміни на цій ітерації
    while (changed) {         // повторюємо поки є що оновлювати
        changed = 0;          // скидаємо прапорець перед кожним проходом

        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                if (edge[y * w + x] != 128) continue;

                int promote = 0;  // прапорець сильного сусіда  

                // перевіряємо всіх  сусідв
                for (int dy = -1; dy <= 1 && !promote; dy++)
                    for (int dx = -1; dx <= 1 && !promote; dx++)
                        if (edge[(y+dy)*w + (x+dx)] == 255) promote = 1; 

                if (promote) {
                    edge[y * w + x] = 255;  
                    changed = 1;
                }
            }
        }
    }

    for (int i = 0; i < n; i++)
        if (edge[i] == 128) edge[i] = 0;  // кандидат без сильних сусідів - фон

    Image *dst = alloc_image(w, h); // повертаємо резалт
    memcpy(dst->data, edge, n); 
    free(edge);
    return dst;
}

static void usage(const char *prog) {//виводить підказку по використанню
    fprintf(stderr,
        "використання:\n"
        "  Режим live (stdin -> stdout):\n"
        "    %s sobel\n"
        "    %s canny\n"
        "  Режим файл:\n"
        "    %s input.ppm output.ppm sobel\n"
        "    %s input.ppm output.ppm canny\n",
        prog, prog, prog, prog);
}
static Image *run_algorithm(const Image *src, const char *mode, //запускає потрібний алгоритм 
                            float low, float high) {
    if (strcmp(mode, "sobel") == 0) {
        return sobel(src);
    } else if (strcmp(mode, "canny") == 0) {
        if (low <= 0 || high <= 0 || low >= high) {
            fprintf(stderr, "Пороги мусять бути: 0 < low < high\n");
            return NULL;
        }
        return canny(src, low, high);
    }
    fprintf(stderr, "Невідомий режим '%s'. Варіанти: sobel, canny\n", mode);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0]); return 1; }
    int live = (strcmp(argv[1], "sobel") == 0 || strcmp(argv[1], "canny") == 0);

    if (!live) {
        if (argc < 4) { usage(argv[0]); return 1; }
        const char *in_path  = argv[1];
        const char *out_path = argv[2];
        const char *mode     = argv[3];
        float low  = (argc > 4) ? (float)atof(argv[4]) : 0.05f;
        float high = (argc > 5) ? (float)atof(argv[5]) : 0.15f;

        Image *src = read_image(in_path);
        if (!src) return 1;
        Image *result = run_algorithm(src, mode, low, high);
        if (result) {
            write_image(out_path, result);
            fprintf(stderr, "Збережено: %s (%dx%d)\n",
                    out_path, result->width, result->height);
            free_image(result);
        }
        free_image(src);
        return result ? 0 : 1;
    }
    const char *mode = argv[1];
    float low  = (argc > 2) ? (float)atof(argv[2]) : 0.05f;
    float high = (argc > 3) ? (float)atof(argv[3]) : 0.15f;

#ifdef _WIN32
    _setmode(_fileno(stdin),  _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    fprintf(stderr, "Vision live: mode=%s", mode);
    if (strcmp(mode, "canny") == 0)
        fprintf(stderr, " low=%.2f high=%.2f", low, high);
    fprintf(stderr, "\n");

    Image *src;
    while ((src = read_image_fp(stdin)) != NULL) {
        Image *result = run_algorithm(src, mode, low, high);
        if (result) {
            Image *frame = make_sidebyside(src, result);
            write_image_fp(stdout, frame);
            free_image(frame);
            free_image(result);
        }
        free_image(src);
    }
    return 0;
}
