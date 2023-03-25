#include <math.h>
#define X_SIZE 1920
#define Y_SIZE 1080

#define PI 3.14149265F

// 波の長さ
#define WAVE_LENGTH 0.000000532F

#define POINT_NUMBERS 11646

// サンプリング周期
#define SAMPLING_INTERVAL 0.000008F

// ホログラム用定数
#define NORM_CONST 40.5845105F

int input();

unsigned char hologram[X_SIZE * Y_SIZE];

float pls_x[POINT_NUMBERS];
float pls_y[POINT_NUMBERS];
float pls_z[POINT_NUMBERS];

float h_r, h_i, temp_h, theta, temp_x, temp_y;

int main() {
  int adr;
  input();
  for (int n = 0; n < Y_SIZE; n++) {
    for (int m = 0; m < X_SIZE; m++) {
      adr = m + n * X_SIZE;
      h_r = 0.0f;
      h_i = 0.0f;
      // 点のループ
      for (int l = 0; l < POINT_NUMBERS; l++) {
        temp_x = pls_x[l] - m * SAMPLING_INTERVAL;
        temp_y = pls_y[l] - n * SAMPLING_INTERVAL;
        // 5.4
        theta = (temp_x * temp_x + temp_y * temp_y) * pls_z[l];
        // 5.2
        h_r += cosf(theta);
        // 5.3
        h_i += sinf(theta);
      }
      // 5.1
      temp_h = atan2f(h_i, h_r);
      if (temp_h < 0.0)
        temp_h += PI * 2;

      // 8bit
      hologram[adr] = (unsigned char)(temp_h * NORM_CONST);
    }
  }
}
