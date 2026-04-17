/*
 * vision.c — Sobel & Canny edge detection на чистому C
 *
 * Компіляція: gcc -O2 -Wall -o vision vision.c -lm
 *   -O2    : оптимізація компілятора (швидше працює)
 *   -Wall  : показує всі попередження
 *   -lm    : лінкуємо math-бібліотеку (для sqrtf, atan2f, тощо)
 *
 * Використання:
 *   ./vision input.ppm output.ppm sobel
 *   ./vision input.ppm output.ppm canny [low_thresh] [high_thresh]
 *
 * Дефолтні пороги Канні: low=0.05, high=0.15 (відносні, від 0.0 до 1.0)
 * Якщо картинка зашумлена — збільши пороги (напр. 0.1 / 0.3)
 */

/* ────────────────────────────────────────────────────────────────
   ПІДКЛЮЧЕННЯ БІБЛІОТЕК
──────────────────────────────────────────────────────────────── */

#include <stdio.h>    /* fopen, fclose, fread, fwrite, fprintf, printf, fgets, sscanf */
#include <stdlib.h>   /* malloc, calloc, free, exit, atof */
#include <string.h>   /* strcmp, strncmp, memcpy */
#include <math.h>     /* sqrtf, atan2f, M_PI */
#include <stdint.h>   /* uint8_t — гарантований 8-бітний беззнаковий тип */
#ifdef _WIN32
#include <fcntl.h>    /* _O_BINARY */
#include <io.h>       /* _setmode, _fileno */
#endif


/* ════════════════════════════════════════════════════════════════
   СТРУКТУРИ
════════════════════════════════════════════════════════════════ */

/*
 * Image — структура що описує одноканальне (grayscale) зображення.
 * Пікселі зберігаємо як плаский одномірний масив:
 *   data[y * width + x] = яскравість пікселя (x, y), від 0 (чорний) до 255 (білий)
 * Це стандартний "row-major" порядок — рядок за рядком зліва направо.
 */
typedef struct {
    int width;       /* ширина зображення в пікселях */
    int height;      /* висота зображення в пікселях */
    uint8_t *data;   /* масив яскравостей, розмір = width * height байт */
} Image;


/* ════════════════════════════════════════════════════════════════
   УТИЛІТИ
════════════════════════════════════════════════════════════════ */

/*
 * clamp_u8 — обрізає float до діапазону [0, 255] і повертає uint8_t.
 * Потрібно скрізь, де рахуємо згортку: результат може вийти за межі.
 * inline — підказка компілятору вставити код прямо на місці виклику (без overhead виклику функції).
 */
static inline uint8_t clamp_u8(float v) {
    if (v < 0.0f)   return 0;    /* від'ємні → чорний */
    if (v > 255.0f) return 255;  /* більше 255 → білий */
    return (uint8_t)v;           /* інакше просто відрізаємо дробову частину */
}

/*
 * idx — перетворює 2D-координати (x, y) в індекс плаского масиву.
 * Формула: y-й рядок починається з позиції y*width, далі +x.
 */
static inline int idx(const Image *img, int x, int y) {
    return y * img->width + x;
}

/*
 * get_pixel — безпечно читає піксель з координатами (x, y).
 * Якщо координата виходить за межу — застосовуємо дзеркальне відбиття.
 * Навіщо: при згортці 3x3 або 5x5 крайні пікселі мають "сусідів" за межею.
 * Варіанти обробки країв:
 *   - нулі (padding zeros) — простіше, але дає темні смуги на краях
 *   - дзеркало (mirror)    — більш природньо, менше артефактів <- наш вибір
 *   - clamp                — повторення крайнього пікселя
 */
static inline uint8_t get_pixel(const Image *img, int x, int y) {
    if (x < 0)            x = -x;                        /* ліва межа: відзеркалюємо */
    if (y < 0)            y = -y;                        /* верхня межа: відзеркалюємо */
    if (x >= img->width)  x = 2 * img->width  - x - 2;  /* права межа: відзеркалюємо */
    if (y >= img->height) y = 2 * img->height - y - 2;  /* нижня межа: відзеркалюємо */
    return img->data[idx(img, x, y)];                    /* повертаємо піксель за індексом */
}

/*
 * alloc_image — виділяє нову Image заданого розміру, заповнену нулями.
 * calloc (на відміну від malloc) ініціалізує пам'ять нулями — важливо,
 * щоб краї / непроініціалізовані пікселі були чорними, а не сміттям.
 */
static Image *alloc_image(int w, int h) {
    Image *img = malloc(sizeof(Image));                    /* виділяємо саму структуру */
    if (!img) { fprintf(stderr, "OOM\n"); exit(1); }      /* OOM = Out Of Memory */
    img->width  = w;                                       /* зберігаємо ширину */
    img->height = h;                                       /* зберігаємо висоту */
    img->data   = calloc((size_t)w * h, 1);                /* w*h байт, всі = 0 */
    if (!img->data) { fprintf(stderr, "OOM\n"); exit(1); }
    return img;
}

/*
 * free_image — звільняє пам'ять Image.
 * Спочатку data (внутрішній масив), потім сама структура.
 * Порядок важливий: після free(img) до img->data вже не можна звертатись.
 */
static void free_image(Image *img) {
    if (img) {           /* захист від free(NULL) — NULL перевіряємо вручну */
        free(img->data); /* звільняємо масив пікселів */
        free(img);       /* звільняємо саму структуру */
    }
}


/* ════════════════════════════════════════════════════════════════
   I/O: PPM (P6, кольоровий) і PGM (P5, сірий)

   Формат PPM/PGM — найпростіший растровий формат без стиснення:
     P6\n           <- magic number: P6=RGB, P5=grayscale
     <width> <height>\n
     255\n          <- максимальне значення (ми завжди пишемо 255)
     <бінарні байти пікселів>
   Для P6: 3 байти на піксель (R, G, B)
   Для P5: 1 байт на піксель (яскравість)
════════════════════════════════════════════════════════════════ */

/*
 * ppm_readline — читає рядок із файлу, пропускаючи рядки-коментарі (#...).
 * Деякі PPM-файли містять коментарі в заголовку — треба їх ігнорувати.
 */
static void ppm_readline(FILE *f, char *buf, int size) {
    do {
        if (!fgets(buf, size, f)) { /* читаємо рядок у буфер */
            buf[0] = '\0';          /* якщо EOF або помилка — повертаємо порожній рядок */
            return;
        }
    } while (buf[0] == '#');        /* повторюємо поки рядок починається з '#' */
}

/*
 * read_image — читає P6 (RGB) або P5 (grayscale) PPM/PGM файл.
 * Якщо P6 — конвертує RGB->grayscale одразу при читанні.
 * Повертає вказівник на нову Image, або NULL при помилці.
 *
 * Формула grayscale (стандарт ITU-R BT.601):
 *   Y = 0.299*R + 0.587*G + 0.114*B
 * Коефіцієнти враховують чутливість ока до різних кольорів
 * (зелений сприймається яскравішим, синій — темнішим).
 */
Image *read_image(const char *path) {
    FILE *f = fopen(path, "rb");  /* "rb" = read binary, важливо на Windows */
    if (!f) {
        fprintf(stderr, "Не можу відкрити '%s'\n", path);
        return NULL;
    }

    char magic[8];                              /* буфер для magic number */
    ppm_readline(f, magic, sizeof(magic));      /* читаємо перший рядок заголовка */

    int is_rgb = 0;                             /* прапорець: 1=RGB, 0=grayscale */
    if      (strncmp(magic, "P6", 2) == 0) is_rgb = 1;   /* P6 = кольоровий PPM */
    else if (strncmp(magic, "P5", 2) == 0) is_rgb = 0;   /* P5 = сірий PGM */
    else {
        fprintf(stderr, "Підтримується тільки P5/P6 PPM/PGM\n");
        fclose(f);
        return NULL;
    }

    char buf[64];
    ppm_readline(f, buf, sizeof(buf));          /* читаємо рядок з розмірами */

    int w, h;
    if (sscanf(buf, "%d %d", &w, &h) != 2) {
        /* Деякі генератори пишуть ширину і висоту на окремих рядках */
        int w2;
        if (sscanf(buf, "%d", &w2) != 1) {
            fprintf(stderr, "Поганий заголовок\n");
            fclose(f);
            return NULL;
        }
        w = w2;
        ppm_readline(f, buf, sizeof(buf));      /* читаємо висоту окремо */
        sscanf(buf, "%d", &h);
    }

    ppm_readline(f, buf, sizeof(buf));          /* читаємо рядок maxval (255) — ігноруємо */

    Image *img = alloc_image(w, h);             /* виділяємо пам'ять під зображення */

    if (is_rgb) {
        uint8_t rgb[3];                         /* тимчасовий буфер для одного RGB-пікселя */
        for (int i = 0; i < w * h; i++) {       /* проходимо по всіх пікселях */
            if (fread(rgb, 1, 3, f) != 3) break;  /* читаємо 3 байти: R, G, B */
            /* конвертуємо RGB -> grayscale за формулою Rec.601 */
            img->data[i] = clamp_u8(0.299f * rgb[0] + 0.587f * rgb[1] + 0.114f * rgb[2]);
        }
    } else {
        /* P5 — одразу читаємо всі пікселі в масив (1 байт = 1 піксель) */
        fread(img->data, 1, (size_t)w * h, f);
    }

    fclose(f);   /* закриваємо файл — більше не потрібен */
    return img;
}

/* read_image_fp — те саме що read_image, але читає з довільного FILE*.
 * Використовується для читання кадрів з stdin у режимі live-відео.
 * Повертає NULL при EOF або помилці заголовка. */
static Image *read_image_fp(FILE *f) {
    char magic[8];
    ppm_readline(f, magic, sizeof(magic));
    if (magic[0] == '\0') return NULL;  /* EOF */

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
    ppm_readline(f, buf, sizeof(buf));  /* maxval — ігноруємо */

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

/*
 * write_image — зберігає Image як P5 PGM (grayscale).
 * Завжди пишемо у grayscale, бо всі наші алгоритми повертають одноканальні зображення.
 * Повертає 1 при успіху, 0 при помилці.
 */
int write_image(const char *path, const Image *img) {
    FILE *f = fopen(path, "wb");       /* "wb" = write binary */
    if (!f) {
        fprintf(stderr, "Не можу записати '%s'\n", path);
        return 0;
    }
    fprintf(f, "P5\n%d %d\n255\n",    /* пишемо текстовий заголовок PGM */
            img->width, img->height);
    fwrite(img->data, 1,               /* пишемо всі байти пікселів одним разом */
           (size_t)img->width * img->height, f);
    fclose(f);
    return 1;
}

/* write_image_fp — те саме що write_image, але пише у довільний FILE*.
 * fflush обов'язковий: без нього кадри застрягають у буфері і ffplay
 * нічого не отримує поки буфер не переповниться. */
static int write_image_fp(FILE *f, const Image *img) {
    fprintf(f, "P5\n%d %d\n255\n", img->width, img->height);
    fwrite(img->data, 1, (size_t)img->width * img->height, f);
    fflush(f);
    return 1;
}

/* make_sidebyside — створює новий Image шириною left->width + right->width.
 * Ліва половина = оригінал (сіре), права = результат алгоритму.
 * Висота = мінімум з двох (на випадок якщо розміри раптом різні). */
static Image *make_sidebyside(const Image *left, const Image *right) {
    int h = left->height < right->height ? left->height : right->height;
    int w = left->width + right->width;
    Image *out = alloc_image(w, h);
    for (int y = 0; y < h; y++) {
        memcpy(out->data + y * w,               left->data  + y * left->width,  left->width);
        memcpy(out->data + y * w + left->width, right->data + y * right->width, right->width);
    }
    return out;
}


/* ════════════════════════════════════════════════════════════════
   GAUSSIAN BLUR

   Розмиваємо зображення ядром Гаусса 5x5 перед Канні.
   Навіщо: пошук контурів дуже чутливий до шуму.
   Один шумний піксель може дати фейковий "контур".
   Розмиття усереднює сусідів -> шум зникає -> алгоритм бачить
   тільки справжні краї.

   Ядро 5x5, sigma ≈ 1.4 (стандартне для Канні):
     Кожне значення = round(255 * G(x,y, sigma)) де G — функція Гаусса
     Нормуємо на суму всіх коефіцієнтів (GAUSS5_SUM = 159)
════════════════════════════════════════════════════════════════ */

/*
 * Ядро Гаусса 5x5. Записане як цілі числа (не поділені на суму) —
 * ділимо при обчисленні щоб уникнути накопичення похибки float.
 */
static const float GAUSS5[5][5] = {
    { 2,  4,  5,  4,  2},   /* верхній рядок (найдальше від центру — найменший вплив) */
    { 4,  9, 12,  9,  4},
    { 5, 12, 15, 12,  5},   /* центральний рядок (найбільші коефіцієнти) */
    { 4,  9, 12,  9,  4},
    { 2,  4,  5,  4,  2},   /* нижній рядок */
};
static const float GAUSS5_SUM = 159.0f; /* сума всіх елементів ядра, для нормалізації */

/*
 * gaussian_blur — повертає нове розмите зображення.
 * Для кожного пікселя (x,y) обчислюємо зважену суму його 5x5-сусідів.
 * Це і є "згортка" (convolution) — базова операція обробки зображень.
 */
static Image *gaussian_blur(const Image *src) {
    Image *dst = alloc_image(src->width, src->height); /* результат того ж розміру */

    for (int y = 0; y < src->height; y++) {       /* рядок за рядком */
        for (int x = 0; x < src->width; x++) {    /* піксель за пікселем */
            float acc = 0.0f;                       /* акумулятор зваженої суми */

            for (int ky = 0; ky < 5; ky++)          /* ky = рядок ядра (0..4) */
                for (int kx = 0; kx < 5; kx++)      /* kx = стовпець ядра (0..4) */
                    /* зміщення -2 центрує ядро на поточному пікселі:
                     * kx=0 -> x-2, kx=1 -> x-1, kx=2 -> x, kx=3 -> x+1, kx=4 -> x+2 */
                    acc += GAUSS5[ky][kx] * get_pixel(src, x + kx - 2, y + ky - 2);

            /* ділимо на суму коефіцієнтів щоб середня яскравість не змінилась */
            dst->data[idx(dst, x, y)] = clamp_u8(acc / GAUSS5_SUM);
        }
    }
    return dst;
}


/* ════════════════════════════════════════════════════════════════
   SOBEL

   Собель знаходить краї (різкі переходи яскравості).
   Ідея: якщо зліва темно, а справа світло — тут є межа.
   Реалізація: дві згортки 3x3.
     Kx шукає вертикальні межі (горизонтальний градієнт dI/dx)
     Ky шукає горизонтальні межі (вертикальний градієнт dI/dy)
   Сила контуру: G = sqrt(Gx^2 + Gy^2)
════════════════════════════════════════════════════════════════ */

/*
 * sobel — повертає карту сили контурів.
 * Більш яскравий піксель у результаті = сильніший контур у джерелі.
 */
Image *sobel(const Image *src) {
    /*
     * Ядро Kx: реагує на вертикальні межі.
     * Ліва колонка від'ємна, права — позитивна.
     * Якщо зліва від пікселя темно (мале значення) а справа світло (велике):
     *   Gx = (-1)*темно + (+1)*світло = велике позитивне число -> є межа.
     *
     *  -1  0  +1
     *  -2  0  +2   <- центральний рядок має подвійну вагу (більш важливий)
     *  -1  0  +1
     */
    static const int Kx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };

    /*
     * Ядро Ky: реагує на горизонтальні межі.
     * Верхній рядок від'ємний, нижній — позитивний.
     *
     *  -1  -2  -1
     *   0   0   0   <- центральний рядок нульовий (не бере участі)
     *  +1  +2  +1
     */
    static const int Ky[3][3] = { {-1,-2,-1}, { 0, 0, 0}, { 1, 2, 1} };

    Image *dst = alloc_image(src->width, src->height); /* результат того ж розміру */

    for (int y = 0; y < src->height; y++) {       /* проходимо по рядках */
        for (int x = 0; x < src->width; x++) {    /* проходимо по стовпцях */
            float gx = 0.0f;  /* компонента горизонтального градієнта */
            float gy = 0.0f;  /* компонента вертикального градієнта */

            for (int ky = 0; ky < 3; ky++)          /* рядок ядра 3x3 */
                for (int kx = 0; kx < 3; kx++) {    /* стовпець ядра 3x3 */
                    float p = get_pixel(src, x + kx - 1, y + ky - 1); /* сусідній піксель */
                    /* -1 зміщує ядро: kx=0 -> x-1, kx=1 -> x, kx=2 -> x+1 */
                    gx += Kx[ky][kx] * p;  /* додаємо внесок у горизонтальний градієнт */
                    gy += Ky[ky][kx] * p;  /* додаємо внесок у вертикальний градієнт */
                }

            /* Евклідова норма градієнта = сила контуру.
             * Теорема Піфагора: G = sqrt(Gx^2 + Gy^2)
             * sqrtf — float-версія sqrt, швидша. */
            dst->data[idx(dst, x, y)] = clamp_u8(sqrtf(gx*gx + gy*gy));
        }
    }
    return dst;
}


/* ════════════════════════════════════════════════════════════════
   CANNY

   Канні — це "прокачаний Собель" з 5 кроками:
     1. Gaussian Blur  — прибрати шум
     2. Sobel          — знайти градієнти (силу + напрямок)
     3. NMS            — стоншити контури до 1 пікселя
     4. Double Thresh  — розділити пікселі на "сильні" / "кандидати" / "сміття"
     5. Hysteresis     — кандидат стає контуром якщо торкається сильного пікселя
════════════════════════════════════════════════════════════════ */

/*
 * quantize_angle — округлює кут градієнта до одного з 4 напрямків:
 *   0   — горизонталь
 *   45  — діагональ вниз-праворуч
 *   90  — вертикаль
 *   135 — діагональ вниз-ліворуч
 *
 * Навіщо: при NMS треба знати, в якому напрямку "рухається" контур,
 * щоб порівняти піксель з його сусідами вздовж цього напрямку.
 * 4 напрямки = 4 пари сусідів для порівняння.
 */
static inline int quantize_angle(float angle_rad) {
    float deg = angle_rad * (180.0f / (float)M_PI); /* радіани -> градуси */
    if (deg < 0) deg += 180.0f;  /* atan2 повертає [-pi, pi], нам потрібно [0, 180] */

    /* Ділимо діапазон [0, 180] на 4 сектори по 45 градусів */
    if      (deg <  22.5f || deg >= 157.5f) return 0;    /* горизонталь */
    else if (deg <  67.5f)                  return 45;   /* діагональ / */
    else if (deg < 112.5f)                  return 90;   /* вертикаль */
    else                                    return 135;  /* діагональ \ */
}

/*
 * canny — повний пайплайн алгоритму Канні.
 *
 * low_t  — нижній поріг (0.0–1.0), нижче = відкидаємо
 * high_t — верхній поріг (0.0–1.0), вище = точно контур
 * Обидва відносні: 0.05 = 5% від максимального градієнта на картинці.
 *
 * Результат: бінарне зображення (255 = контур, 0 = фон).
 */
Image *canny(const Image *src, float low_t, float high_t) {
    int w = src->width;    /* ширина */
    int h = src->height;   /* висота */
    int n = w * h;         /* загальна кількість пікселів */

    /* ── КРОК 1: Gaussian Blur ─────────────────────────────────
     * Розмиваємо щоб позбутись шуму перед знаходженням градієнтів.
     * Без цього кожен шумний піксель дасть фейковий "контур". */
    Image *blurred = gaussian_blur(src);

    /* ── КРОК 2: Sobel (з float-точністю) ──────────────────────
     * Рахуємо градієнти у float (не uint8_t!) щоб:
     *   а) не втратити точність при NMS
     *   б) мати кути для визначення напрямку контуру */

    /* Ті самі ядра Собеля що і у функції sobel() вище */
    static const int Kx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    static const int Ky[3][3] = { {-1,-2,-1}, { 0, 0, 0}, { 1, 2, 1} };

    float *magnitude = malloc(n * sizeof(float));  /* сила градієнта для кожного пікселя */
    int   *direction = malloc(n * sizeof(int));    /* квантований напрямок (0/45/90/135) */
    if (!magnitude || !direction) { fprintf(stderr, "OOM\n"); exit(1); }

    float max_mag = 0.0f;  /* максимальна сила градієнта на всій картинці */

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float gx = 0.0f, gy = 0.0f;   /* горизонтальний і вертикальний градієнти */

            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++) {
                    float p = get_pixel(blurred, x + kx - 1, y + ky - 1); /* беремо з розмитого! */
                    gx += Kx[ky][kx] * p;  /* внесок у горизонтальний градієнт */
                    gy += Ky[ky][kx] * p;  /* внесок у вертикальний градієнт */
                }

            float mag = sqrtf(gx*gx + gy*gy);    /* сила контуру в цій точці */
            magnitude[y * w + x] = mag;            /* зберігаємо силу для NMS */

            /* atan2f(gy, gx) = кут вектора градієнта в радіанах */
            direction[y * w + x] = quantize_angle(atan2f(gy, gx)); /* зберігаємо напрямок */

            if (mag > max_mag) max_mag = mag;      /* оновлюємо максимум для нормалізації порогів */
        }
    }

    free_image(blurred);  /* розмите зображення більше не потрібне — звільняємо */

    /* Перетворюємо відносні пороги (0.0–1.0) в абсолютні значення градієнта.
     * Наприклад: low_t=0.05, max_mag=500 -> low=25.0 */
    float low  = low_t  * max_mag;
    float high = high_t * max_mag;

    /* ── КРОК 3: Non-Maximum Suppression (NMS) ──────────────────
     * Проблема після Собеля: контури "товсті" (3-5 пікселів шириною).
     * NMS залишає тільки локальні максимуми вздовж напрямку градієнта.
     * Логіка: якщо піксель не є найяскравішим серед своїх двох сусідів
     *         у напрямку градієнта — він не справжній край, обнуляємо. */

    float *nms = calloc(n, sizeof(float));  /* результат NMS, спочатку всі нулі */
    if (!nms) { fprintf(stderr, "OOM\n"); exit(1); }

    /* Пропускаємо крайові пікселі (y=0, y=h-1, x=0, x=w-1):
     * у них немає повного набору сусідів для порівняння */
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            float m = magnitude[y * w + x];  /* сила поточного пікселя */
            float a, b;  /* два сусіди вздовж напрямку градієнта */

            /* Вибираємо пару сусідів залежно від напрямку контуру:
             *    0  -> сусіди зліва/справа      (горизонтальний контур)
             *   45  -> сусіди по діагоналі
             *   90  -> сусіди зверху/знизу       (вертикальний контур)
             *  135  -> сусіди по іншій діагоналі */
            switch (direction[y * w + x]) {
                case   0: a = magnitude[y * w + x - 1];       /* ліво */
                          b = magnitude[y * w + x + 1];       /* право */
                          break;
                case  45: a = magnitude[(y-1)*w + x+1];       /* вверх-право */
                          b = magnitude[(y+1)*w + x-1];       /* вниз-ліво */
                          break;
                case  90: a = magnitude[(y-1)*w + x];         /* вверх */
                          b = magnitude[(y+1)*w + x];         /* вниз */
                          break;
                default:  a = magnitude[(y-1)*w + x-1];       /* вверх-ліво */
                          b = magnitude[(y+1)*w + x+1];       /* вниз-право */
                          break;
            }

            /* Залишаємо піксель тільки якщо він >= обох сусідів.
             * Якщо менший хоча б одного — це не локальний максимум -> 0. */
            nms[y * w + x] = (m >= a && m >= b) ? m : 0.0f;
        }
    }

    free(magnitude);  /* magnitude більше не потрібна */
    free(direction);  /* direction більше не потрібна */

    /* ── КРОК 4: Double Thresholding ────────────────────────────
     * Класифікуємо кожен піксель на 3 категорії:
     *   255 (STRONG)    — точно контур (mag >= high)
     *   128 (CANDIDATE) — може бути контуром (low <= mag < high)
     *     0 (WEAK)      — точно не контур (mag < low), відкидаємо */

    uint8_t *edge = calloc(n, 1);  /* масив міток, спочатку всі = 0 (WEAK) */
    if (!edge) { fprintf(stderr, "OOM\n"); exit(1); }

    for (int i = 0; i < n; i++) {
        if      (nms[i] >= high) edge[i] = 255;  /* сильний -> точно контур */
        else if (nms[i] >= low)  edge[i] = 128;  /* кандидат -> може бути контуром */
        /* інакше залишається 0 — відкидаємо */
    }

    free(nms);  /* nms більше не потрібна */

    /* ── КРОК 5: Edge Tracking by Hysteresis ────────────────────
     * Кандидат (128) стає сильним (255) якщо має сильного сусіда.
     * Ідея: справжній контур — неперервна лінія. Якщо кандидат "тягнеться"
     *       від сильного пікселя — він частина контуру. Інакше — шум.
     *
     * Реалізація: ітеративно проходимо по всіх кандидатах.
     * Якщо хоч один з 8 сусідів = 255 -> підвищуємо кандидата до 255.
     * Повторюємо поки є зміни (поки "вогонь" розповсюджується).
     *
     * Чому ітеративно а не рекурсивно (flood fill)?
     * Рекурсія на великій картинці може переповнити стек (stack overflow).
     * При розмірі 4000x4000 і поганому випадку стек може сягнути мільйони кадрів.
     * Ітеративний підхід — безпечний, без обмежень на глибину. */

    int changed = 1;          /* прапорець: чи були зміни на цій ітерації */
    while (changed) {         /* повторюємо поки є що оновлювати */
        changed = 0;          /* скидаємо прапорець перед кожним проходом */

        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                if (edge[y * w + x] != 128) continue;  /* нас цікавлять тільки кандидати */

                int promote = 0;  /* прапорець: знайшли сильного сусіда? */

                /* Перевіряємо всіх 8 сусідів (dy, dx належать {-1, 0, 1}) */
                for (int dy = -1; dy <= 1 && !promote; dy++)
                    for (int dx = -1; dx <= 1 && !promote; dx++)
                        if (edge[(y+dy)*w + (x+dx)] == 255) promote = 1; /* знайшли сильного! */

                if (promote) {
                    edge[y * w + x] = 255;  /* кандидат -> сильний */
                    changed = 1;             /* щось змінилось -> потрібна ще одна ітерація */
                }
            }
        }
    }

    /* Всі кандидати що не підвищились до 255 — це ізольований шум, вбиваємо */
    for (int i = 0; i < n; i++)
        if (edge[i] == 128) edge[i] = 0;  /* кандидат без сильних сусідів -> фон */

    /* Пакуємо результат у Image і повертаємо */
    Image *dst = alloc_image(w, h);
    memcpy(dst->data, edge, n);  /* копіюємо n байт з edge -> dst->data */
    free(edge);                   /* тимчасовий масив більше не потрібен */
    return dst;
}


/* ════════════════════════════════════════════════════════════════
   MAIN / CLI
════════════════════════════════════════════════════════════════ */

/*
 * usage — виводить підказку по використанню і виходить.
 * argv[0] = ім'я програми (наприклад "./vision")
 */
static void usage(const char *prog) {
    fprintf(stderr,
        "Використання:\n"
        "  Режим live (stdin -> stdout):\n"
        "    %s sobel\n"
        "    %s canny [low=0.05] [high=0.15]\n"
        "  Режим файл:\n"
        "    %s input.ppm output.ppm sobel\n"
        "    %s input.ppm output.ppm canny [low=0.05] [high=0.15]\n",
        prog, prog, prog, prog);
}

/* run_algorithm — запускає потрібний алгоритм на src і повертає результат. */
static Image *run_algorithm(const Image *src, const char *mode,
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

    /* Визначаємо режим: якщо перший аргумент — назва алгоритму, то live-режим.
     * Інакше перший аргумент — шлях до файлу (старий режим). */
    int live = (strcmp(argv[1], "sobel") == 0 || strcmp(argv[1], "canny") == 0);

    if (!live) {
        /* ── РЕЖИМ ФАЙЛ ─────────────────────────────────────────
         * ./vision input.ppm output.ppm sobel
         * ./vision input.ppm output.ppm canny [low] [high]      */
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

    /* ── РЕЖИМ LIVE ─────────────────────────────────────────────
     * ffmpeg ... - | ./vision sobel | ffplay ...
     * Читаємо PPM-кадри з stdin, обробляємо, пишемо PGM side-by-side у stdout.
     * Виводимо подвійно широкий кадр: ліворуч оригінал, праворуч результат. */
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
