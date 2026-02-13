#include <initializer_list>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <jni.h>
#include <cerrno>
#include <cassert>
#include <EGL/egl.h>
#include <GLES/gl.h>
#include <GLES3/gl3.h>
#include <GLES3/gl3ext.h>
#include <android/sensor.h>
#include <android/log.h>
#include <android_native_app_glue.h>
#include <time.h>
#include <math.h>
#include <dlfcn.h>
#include "../utils/utils.h"  // 保留你的工具头文件（如有）

#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "native-activity", __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, "native-activity", __VA_ARGS__))

// ---------- 粒子系统常量 ----------
#define MAX_PARTICLES        8000
#define PARTICLE_LIFETIME    2.0f
#define ROCKET_LIFETIME      1.8f
#define EXPLOSION_COUNT      35
#define TRAIL_COUNT          3     // 每个火箭每帧产生尾迹粒子数
#define FIREWORK_COOLDOWN    0.25f // 生成火箭的间隔
#define TEXT_PARTICLE_COUNT  1200  // 文字爆炸粒子数

// ---------- 粒子类型 ----------
enum ParticleType {
    PARTICLE_ROCKET,
    PARTICLE_EXPLOSION,
    PARTICLE_TRAIL,
    PARTICLE_TEXT
};

// ---------- 粒子结构 ----------
typedef struct Particle {
    float x, y;          // 位置
    float vx, vy;        // 速度
    float ax, ay;        // 加速度
    float r, g, b, a;    // 颜色
    float size;          // 大小
    float life;          // 剩余生命
    float maxLife;       // 最大生命
    int type;            // 粒子类型
} Particle;

// ---------- 保存的状态（兼容原有结构） ----------
struct saved_state {
    float angle;
    int32_t x;
    int32_t y;
    struct timeval startTime;
};

// ---------- OpenGL 资源结构 ----------
struct glstruct {
    GLuint program;      // 粒子着色器程序
    GLuint vertexShader;
    GLuint fragmentShader;
    GLuint texture;      // 粒子纹理（圆形渐变）
    GLint  uMatrix;
    GLint  uTexture;
    GLint  aPosition;
    GLint  aColor;
    GLint  aSize;
};

// ---------- 引擎主结构 ----------
struct engine {
    struct android_app *app;
    ASensorManager *sensorManager;
    const ASensor *accelerometerSensor;
    ASensorEventQueue *sensorEventQueue;
    int animating;
    EGLDisplay display;
    EGLSurface surface;
    EGLContext context;
    int32_t width;
    int32_t height;
    struct saved_state state;
    struct glstruct gldata;

    // 粒子系统数据
    Particle particles[MAX_PARTICLES];
    int particleCount;
    float fireworkTimer;       // 火箭发射计时器
    float totalTime;          // 总运行时间
    int   textSpawned;        // 文字是否已生成
    float exitTimer;         // 退出倒计时
    int   shouldExit;        // 是否需要退出
};

// ---------- 正交投影矩阵（工具函数，保留） ----------
static void orthoM(float* m, int offset, float left, float right, float bottom, float top, float near, float far) {
    m[offset + 0] = 2 / (right - left);
    m[offset + 1] = 0;
    m[offset + 2] = 0;
    m[offset + 3] = 0;
    m[offset + 4] = 0;
    m[offset + 5] = 2 / (top - bottom);
    m[offset + 6] = 0;
    m[offset + 7] = 0;
    m[offset + 8] = 0;
    m[offset + 9] = 0;
    m[offset + 10] = -2 / (far - near);
    m[offset + 11] = 0;
    m[offset + 12] = -(right + left) / (right - left);
    m[offset + 13] = -(top + bottom) / (top - bottom);
    m[offset + 14] = -(far + near) / (far - near);
    m[offset + 15] = 1;
}

// ---------- 着色器编译检查 ----------
static void checkShaderCompile(GLuint shader) {
    GLint compiled;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    GLint infoLen = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);
    if (!compiled) {
        if (infoLen > 1) {
            char *infoLog = (char *) malloc(sizeof(char) * infoLen);
            glGetShaderInfoLog(shader, infoLen, NULL, infoLog);
            LOGW("Error compiling shader:[%s]", infoLog);
            free(infoLog);
        }
        glDeleteShader(shader);
    } else {
        LOGI("Shader compiled successfully");
    }
}

// ---------- 创建圆形渐变纹理 ----------
static GLuint createCircleTexture() {
    const int texSize = 64;
    unsigned char data[texSize * texSize * 4];
    for (int y = 0; y < texSize; y++) {
        for (int x = 0; x < texSize; x++) {
            float dx = (x + 0.5f - texSize / 2) / (texSize / 2);
            float dy = (y + 0.5f - texSize / 2) / (texSize / 2);
            float d = sqrtf(dx * dx + dy * dy);
            float alpha = 1.0f - (d > 1.0f ? 1.0f : d);
            alpha = alpha * alpha; // 更平滑
            int idx = (y * texSize + x) * 4;
            data[idx] = 255;
            data[idx+1] = 255;
            data[idx+2] = 255;
            data[idx+3] = (unsigned char)(alpha * 255);
        }
    }
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texSize, texSize, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return tex;
}

// ---------- 初始化粒子着色器 ----------
static int init_particle_shader(struct engine *engine) {
    // 顶点着色器：仅传递位置、颜色、点大小，并计算点精灵坐标
    const char* vertexShaderSrc =
        "#version 300 es\n"
        "uniform mat4 uMatrix;\n"
        "layout(location = 0) in vec2 aPosition;\n"
        "layout(location = 1) in vec4 aColor;\n"
        "layout(location = 2) in float aSize;\n"
        "out vec4 vColor;\n"
        "void main() {\n"
        "    gl_Position = uMatrix * vec4(aPosition, 0.0, 1.0);\n"
        "    gl_PointSize = aSize;\n"
        "    vColor = aColor;\n"
        "}\n";

    // 片段着色器：采样圆形纹理，应用颜色
    const char* fragmentShaderSrc =
        "#version 300 es\n"
        "precision mediump float;\n"
        "uniform sampler2D uTexture;\n"
        "in vec4 vColor;\n"
        "out vec4 fragColor;\n"
        "void main() {\n"
        "    vec4 texColor = texture(uTexture, gl_PointCoord);\n"
        "    fragColor = vColor * texColor;\n"
        "}\n";

    GLuint program = glCreateProgram();
    GLuint vert = glCreateShader(GL_VERTEX_SHADER);
    GLuint frag = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vert, 1, &vertexShaderSrc, NULL);
    glCompileShader(vert);
    checkShaderCompile(vert);

    glShaderSource(frag, 1, &fragmentShaderSrc, NULL);
    glCompileShader(frag);
    checkShaderCompile(frag);

    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);

    GLint linked;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        LOGW("Particle program link failed");
        return -1;
    }

    engine->gldata.program = program;
    engine->gldata.vertexShader = vert;
    engine->gldata.fragmentShader = frag;
    engine->gldata.uMatrix = glGetUniformLocation(program, "uMatrix");
    engine->gldata.uTexture = glGetUniformLocation(program, "uTexture");
    engine->gldata.aPosition = 0;
    engine->gldata.aColor = 1;
    engine->gldata.aSize = 2;

    // 创建纹理
    engine->gldata.texture = createCircleTexture();

    return 0;
}

// ---------- 初始化显示 ----------
static int engine_init_display(struct engine *engine) {
    const EGLint attribs[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_BLUE_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_RED_SIZE, 8,
        EGL_NONE
    };
    EGLint w, h, format;
    EGLint numConfigs;
    EGLConfig config = nullptr;
    EGLSurface surface;
    EGLContext context;
    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (!eglInitialize(display, nullptr, nullptr)) {
        LOGW("egl init failed");
        return 0;
    }

    eglChooseConfig(display, attribs, nullptr, 0, &numConfigs);
    std::unique_ptr<EGLConfig[]> supportedConfigs(new EGLConfig[numConfigs]);
    eglChooseConfig(display, attribs, supportedConfigs.get(), numConfigs, &numConfigs);
    int i = 0;
    for (; i < numConfigs; i++) {
        EGLConfig cfg = supportedConfigs[i];
        EGLint r, g, b, d;
        if (eglGetConfigAttrib(display, cfg, EGL_RED_SIZE, &r) &&
            eglGetConfigAttrib(display, cfg, EGL_GREEN_SIZE, &g) &&
            eglGetConfigAttrib(display, cfg, EGL_BLUE_SIZE, &b) &&
            eglGetConfigAttrib(display, cfg, EGL_DEPTH_SIZE, &d) &&
            r == 8 && g == 8 && b == 8 && d == 0) {
            config = cfg;
            break;
        }
    }
    if (i == numConfigs) config = supportedConfigs[0];
    if (config == nullptr) {
        LOGW("no EGLConfig");
        return -1;
    }

    eglGetConfigAttrib(display, config, EGL_NATIVE_VISUAL_ID, &format);
    surface = eglCreateWindowSurface(display, config, engine->app->window, nullptr);
    EGLint contextAttribs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
    context = eglCreateContext(display, EGL_NO_CONTEXT, EGL_NO_CONTEXT, contextAttribs);
    eglMakeCurrent(display, surface, surface, context);

    eglQuerySurface(display, surface, EGL_WIDTH, &w);
    eglQuerySurface(display, surface, EGL_HEIGHT, &h);

    engine->display = display;
    engine->context = context;
    engine->surface = surface;
    engine->width = w;
    engine->height = h;
    engine->state.angle = 0;

    // 初始化粒子着色器
    if (init_particle_shader(engine) != 0) {
        LOGW("init particle shader failed");
        return -1;
    }

    // 设置OpenGL状态
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    LOGI("OpenGL Info: %s", glGetString(GL_VERSION));
    return 0;
}

// ---------- 粒子系统函数 ----------

// 添加粒子到数组
static void add_particle(struct engine *engine, Particle p) {
    if (engine->particleCount >= MAX_PARTICLES) {
        // 简单替换最老的粒子
        int oldestIdx = 0;
        float oldestLife = engine->particles[0].life;
        for (int i = 1; i < MAX_PARTICLES; i++) {
            if (engine->particles[i].life < oldestLife) {
                oldestLife = engine->particles[i].life;
                oldestIdx = i;
            }
        }
        engine->particles[oldestIdx] = p;
    } else {
        engine->particles[engine->particleCount++] = p;
    }
}

// 生成一枚上升火箭
static void spawn_rocket(struct engine *engine) {
    Particle r;
    float aspect = (float)engine->width / engine->height;
    r.x = ((float)rand() / RAND_MAX) * 2.0f * aspect - aspect; // 屏幕宽度范围
    r.y = -1.0f; // 底部
    r.vx = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    r.vy = ((float)rand() / RAND_MAX) * 0.8f + 0.8f; // 向上速度
    r.ax = 0.0f;
    r.ay = 0.2f; // 减速（模拟重力）
    r.r = ((float)rand() / RAND_MAX) * 0.5f + 0.5f;
    r.g = ((float)rand() / RAND_MAX) * 0.3f + 0.7f;
    r.b = ((float)rand() / RAND_MAX) * 0.2f + 0.8f;
    r.a = 1.0f;
    r.size = 8.0f;
    r.life = ROCKET_LIFETIME;
    r.maxLife = ROCKET_LIFETIME;
    r.type = PARTICLE_ROCKET;
    add_particle(engine, r);
}

// 火箭爆炸，生成爆炸粒子
static void explode(struct engine *engine, float x, float y, float r, float g, float b) {
    for (int i = 0; i < EXPLOSION_COUNT; i++) {
        Particle p;
        p.x = x;
        p.y = y;
        float angle = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
        float speed = ((float)rand() / RAND_MAX) * 1.5f + 0.5f;
        p.vx = cosf(angle) * speed;
        p.vy = sinf(angle) * speed;
        p.ax = 0.0f;
        p.ay = 0.5f; // 重力
        p.r = r;
        p.g = g;
        p.b = b;
        p.a = 1.0f;
        p.size = ((float)rand() / RAND_MAX) * 6.0f + 4.0f;
        p.life = ((float)rand() / RAND_MAX) * 1.5f + 0.8f;
        p.maxLife = p.life;
        p.type = PARTICLE_EXPLOSION;
        add_particle(engine, p);
    }
}

// 生成尾迹粒子（火箭拖尾）
static void spawn_trail(struct engine *engine, float x, float y, float r, float g, float b) {
    for (int i = 0; i < TRAIL_COUNT; i++) {
        Particle t;
        t.x = x + ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
        t.y = y + ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
        t.vx = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        t.vy = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
        t.ax = 0.0f;
        t.ay = 0.1f;
        t.r = r;
        t.g = g;
        t.b = b;
        t.a = 0.7f;
        t.size = ((float)rand() / RAND_MAX) * 4.0f + 2.0f;
        t.life = 0.4f;
        t.maxLife = 0.4f;
        t.type = PARTICLE_TRAIL;
        add_particle(engine, t);
    }
}

// ---------- 生成文字“新年快乐”的粒子（爆炸效果）----------
static void spawn_text_particles(struct engine *engine) {
    // 定义文字轮廓的点阵（归一化坐标 0-1）
    // 简化：四个字“新年快乐”用点阵粗略表示
    // 每个字大约 0.2x0.3，四个字横向排列
    const int pointsPerChar = 300;
    float aspect = (float)engine->width / engine->height;
    for (int i = 0; i < TEXT_PARTICLE_COUNT; i++) {
        Particle p;
        // 随机选择四个字中的一个
        int charIdx = rand() % 4;
        float baseX = -0.6f + charIdx * 0.4f; // 大致位置
        float baseY = 0.0f;

        // 在字的区域内随机偏移
        float cx = ((float)rand() / RAND_MAX) * 0.25f - 0.125f;
        float cy = ((float)rand() / RAND_MAX) * 0.35f - 0.175f;

        // 为不同字添加简单轮廓形状
        if (charIdx == 0) { // 新
            cx = sinf(cx * 10) * 0.1f; // 简单装饰
        } else if (charIdx == 1) { // 年
            cy = cosf(cy * 8) * 0.08f;
        } else if (charIdx == 2) { // 快
            cx = fabsf(cx) - 0.05f;
        } else { // 乐
            cy = fabsf(cy) - 0.05f;
        }

        p.x = (baseX + cx) * aspect; // 考虑屏幕比例
        p.y = baseY + cy;

        // 爆炸速度向外
        float angle = atan2f(p.y - baseY, p.x - baseX);
        float speed = ((float)rand() / RAND_MAX) * 1.0f + 0.3f;
        p.vx = cosf(angle) * speed * 0.5f;
        p.vy = sinf(angle) * speed * 0.5f;
        p.ax = 0.0f;
        p.ay = 0.2f; // 轻微重力

        // 多彩颜色
        p.r = ((float)rand() / RAND_MAX) * 0.8f + 0.2f;
        p.g = ((float)rand() / RAND_MAX) * 0.8f + 0.2f;
        p.b = ((float)rand() / RAND_MAX) * 0.8f + 0.2f;
        p.a = 1.0f;
        p.size = ((float)rand() / RAND_MAX) * 12.0f + 6.0f;
        p.life = ((float)rand() / RAND_MAX) * 2.0f + 1.0f;
        p.maxLife = p.life;
        p.type = PARTICLE_TEXT;
        add_particle(engine, p);
    }
    LOGI("文字粒子已生成！");
}

// ---------- 更新粒子 ----------
static void update_particles(struct engine *engine, float deltaTime) {
    for (int i = engine->particleCount - 1; i >= 0; i--) {
        Particle *p = &engine->particles[i];
        p->life -= deltaTime;
        if (p->life <= 0.0f) {
            // 移除粒子
            engine->particles[i] = engine->particles[--engine->particleCount];
            continue;
        }

        // 物理更新
        p->vx += p->ax * deltaTime;
        p->vy += p->ay * deltaTime;
        p->x += p->vx * deltaTime;
        p->y += p->vy * deltaTime;

        // 火箭特殊处理：到达顶部或生命周期结束时爆炸
        if (p->type == PARTICLE_ROCKET) {
            // 产生尾迹
            spawn_trail(engine, p->x, p->y, p->r, p->g, p->b);
            // 如果超出顶部或生命快结束，爆炸
            if (p->y > 1.2f || p->life < 0.2f) {
                explode(engine, p->x, p->y, p->r, p->g, p->b);
                p->life = 0.0f; // 标记删除
            }
        }

        // 根据生命调整透明度
        p->a = p->life / p->maxLife;
        if (p->type == PARTICLE_TRAIL) {
            p->a *= 0.8f;
            p->size *= 0.95f;
        }
    }
}

// ---------- 渲染所有粒子 ----------
static void render_particles(struct engine *engine) {
    glUseProgram(engine->gldata.program);

    // 设置投影矩阵（保持宽高比）
    float proj[16];
    float aspect = (float)engine->width / engine->height;
    orthoM(proj, 0, -aspect, aspect, -1.0f, 1.0f, -1.0f, 1.0f);
    glUniformMatrix4fv(engine->gldata.uMatrix, 1, GL_FALSE, proj);

    // 绑定纹理
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, engine->gldata.texture);
    glUniform1i(engine->gldata.uTexture, 0);

    // 准备顶点数据
    GLfloat *vertices = (GLfloat*)malloc(engine->particleCount * 7 * sizeof(GLfloat)); // 每粒子: x,y,r,g,b,a,size
    int idx = 0;
    for (int i = 0; i < engine->particleCount; i++) {
        Particle *p = &engine->particles[i];
        vertices[idx++] = p->x;
        vertices[idx++] = p->y;
        vertices[idx++] = p->r;
        vertices[idx++] = p->g;
        vertices[idx++] = p->b;
        vertices[idx++] = p->a;
        vertices[idx++] = p->size;
    }

    glVertexAttribPointer(engine->gldata.aPosition, 2, GL_FLOAT, GL_FALSE, 7 * sizeof(GLfloat), vertices);
    glVertexAttribPointer(engine->gldata.aColor, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(GLfloat), vertices + 2);
    glVertexAttribPointer(engine->gldata.aSize, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(GLfloat), vertices + 6);

    glEnableVertexAttribArray(engine->gldata.aPosition);
    glEnableVertexAttribArray(engine->gldata.aColor);
    glEnableVertexAttribArray(engine->gldata.aSize);

    glDrawArrays(GL_POINTS, 0, engine->particleCount);

    glDisableVertexAttribArray(engine->gldata.aPosition);
    glDisableVertexAttribArray(engine->gldata.aColor);
    glDisableVertexAttribArray(engine->gldata.aSize);

    free(vertices);
}

// ---------- 绘制帧 ----------
static void engine_draw_frame(struct engine *engine) {
    if (engine->display == nullptr) return;

    // 计算时间差
    static struct timeval lastTime = {0, 0};
    struct timeval now;
    gettimeofday(&now, NULL);
    float deltaTime = 0.016f; // 默认约60fps
    if (lastTime.tv_sec != 0) {
        deltaTime = (now.tv_sec - lastTime.tv_sec) + (now.tv_usec - lastTime.tv_usec) * 0.000001f;
        if (deltaTime > 0.1f) deltaTime = 0.016f; // 防止跳帧
    }
    lastTime = now;

    // 总运行时间
    engine->totalTime += deltaTime;

    // 退出逻辑：文字出现后2秒退出
    if (engine->textSpawned) {
        engine->exitTimer -= deltaTime;
        if (engine->exitTimer <= 0.0f && !engine->shouldExit) {
            engine->shouldExit = 1;
            ANativeActivity_finish(engine->app->activity);
            LOGI("2秒已过，退出程序");
        }
    }

    // 10秒触发文字爆炸
    if (!engine->textSpawned && engine->totalTime >= 10.0f) {
        // 清空大部分粒子，保留文字粒子
        engine->particleCount = 0;
        spawn_text_particles(engine);
        engine->textSpawned = 1;
        engine->exitTimer = 2.0f; // 再过2秒退出
    }

    // 生成新的火箭（如果不处于文字阶段）
    if (!engine->textSpawned) {
        engine->fireworkTimer -= deltaTime;
        while (engine->fireworkTimer <= 0.0f) {
            spawn_rocket(engine);
            engine->fireworkTimer += FIREWORK_COOLDOWN;
        }
    }

    // 更新粒子
    update_particles(engine, deltaTime);

    // 渲染
    glClear(GL_COLOR_BUFFER_BIT);
    render_particles(engine);

    eglSwapBuffers(engine->display, engine->surface);
}

// ---------- 终止显示 ----------
static void engine_term_display(struct engine *engine) {
    if (engine->display != EGL_NO_DISPLAY) {
        eglMakeCurrent(engine->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (engine->context != EGL_NO_CONTEXT) {
            eglDestroyContext(engine->display, engine->context);
        }
        if (engine->surface != EGL_NO_SURFACE) {
            eglDestroySurface(engine->display, engine->surface);
        }
        eglTerminate(engine->display);
    }
    engine->animating = 0;
    engine->display = EGL_NO_DISPLAY;
    engine->context = EGL_NO_CONTEXT;
    engine->surface = EGL_NO_SURFACE;
}

// ---------- 输入处理 ----------
static int32_t engine_handle_input(struct android_app *app, AInputEvent *event) {
    auto *engine = (struct engine *) app->userData;
    if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION) {
        engine->animating = 1;
        engine->state.x = AMotionEvent_getX(event, 0);
        engine->state.y = AMotionEvent_getY(event, 0);
        return 1;
    }
    return 0;
}

// ---------- 命令处理 ----------
static void engine_handle_cmd(struct android_app *app, int32_t cmd) {
    auto *engine = (struct engine *) app->userData;
    switch (cmd) {
        case APP_CMD_SAVE_STATE:
            engine->app->savedState = malloc(sizeof(struct saved_state));
            *((struct saved_state *) engine->app->savedState) = engine->state;
            engine->app->savedStateSize = sizeof(struct saved_state);
            break;
        case APP_CMD_INIT_WINDOW:
            if (engine->app->window != nullptr) {
                engine_init_display(engine);
                engine_draw_frame(engine);
            }
            break;
        case APP_CMD_TERM_WINDOW:
            engine_term_display(engine);
            break;
        case APP_CMD_GAINED_FOCUS:
            if (engine->accelerometerSensor != nullptr) {
                ASensorEventQueue_enableSensor(engine->sensorEventQueue, engine->accelerometerSensor);
                ASensorEventQueue_setEventRate(engine->sensorEventQueue, engine->accelerometerSensor, (1000L / 60) * 1000);
            }
            break;
        case APP_CMD_LOST_FOCUS:
            if (engine->accelerometerSensor != nullptr) {
                ASensorEventQueue_disableSensor(engine->sensorEventQueue, engine->accelerometerSensor);
            }
            engine->animating = 0;
            engine_draw_frame(engine);
            break;
        default:
            break;
    }
}

// ---------- 传感器Manager获取（保持不变）----------
ASensorManager *AcquireASensorManagerInstance(android_app *app) {
    if (!app) return nullptr;
    typedef ASensorManager *(*PF_GETINSTANCEFORPACKAGE)(const char *name);
    void *androidHandle = dlopen("libandroid.so", RTLD_NOW);
    auto getInstanceForPackageFunc = (PF_GETINSTANCEFORPACKAGE) dlsym(androidHandle, "ASensorManager_getInstanceForPackage");
    if (getInstanceForPackageFunc) {
        JNIEnv *env = nullptr;
        app->activity->vm->AttachCurrentThread(&env, nullptr);
        jclass android_content_Context = env->GetObjectClass(app->activity->clazz);
        jmethodID midGetPackageName = env->GetMethodID(android_content_Context, "getPackageName", "()Ljava/lang/String;");
        auto packageName = (jstring) env->CallObjectMethod(app->activity->clazz, midGetPackageName);
        const char *nativePackageName = env->GetStringUTFChars(packageName, nullptr);
        ASensorManager *mgr = getInstanceForPackageFunc(nativePackageName);
        env->ReleaseStringUTFChars(packageName, nativePackageName);
        app->activity->vm->DetachCurrentThread();
        if (mgr) {
            dlclose(androidHandle);
            return mgr;
        }
    }
    typedef ASensorManager *(*PF_GETINSTANCE)();
    auto getInstanceFunc = (PF_GETINSTANCE) dlsym(androidHandle, "ASensorManager_getInstance");
    assert(getInstanceFunc);
    dlclose(androidHandle);
    return getInstanceFunc();
}

// ---------- 主入口 ----------
void android_main(struct android_app *state) {
    struct engine engine{};
    memset(&engine, 0, sizeof(engine));

    state->userData = &engine;
    state->onAppCmd = engine_handle_cmd;
    state->onInputEvent = engine_handle_input;
    engine.app = state;

    // 初始化传感器（可选）
    engine.sensorManager = AcquireASensorManagerInstance(state);
    engine.accelerometerSensor = ASensorManager_getDefaultSensor(engine.sensorManager, ASENSOR_TYPE_ACCELEROMETER);
    engine.sensorEventQueue = ASensorManager_createEventQueue(engine.sensorManager, state->looper, LOOPER_ID_USER, nullptr, nullptr);

    if (state->savedState != nullptr) {
        engine.state = *(struct saved_state *) state->savedState;
    }

    // 初始化随机种子
    srand(time(NULL));

    // 记录开始时间
    gettimeofday(&engine.state.startTime, NULL);
    engine.totalTime = 0.0f;
    engine.fireworkTimer = 0.0f;
    engine.textSpawned = 0;
    engine.shouldExit = 0;
    engine.particleCount = 0;

    // 主循环
    while (true) {
        int ident;
        int events;
        struct android_poll_source *source;

        while ((ident = ALooper_pollOnce(engine.animating ? 0 : -1, nullptr, &events, (void **) &source)) >= 0) {
            if (source != nullptr) {
                source->process(state, source);
            }
            if (ident == LOOPER_ID_USER) {
                if (engine.accelerometerSensor != nullptr) {
                    ASensorEvent event;
                    ASensorEventQueue_getEvents(engine.sensorEventQueue, &event, 1);
                }
            }
            if (state->destroyRequested != 0) {
                engine_term_display(&engine);
                return;
            }
        }

        if (engine.animating) {
            engine_draw_frame(&engine);
        }
    }
}