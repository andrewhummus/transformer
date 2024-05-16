#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define MAX_SEQ_LEN 512
#define NUN_HEADS 8
#define D_MODEL 512

void softmax(float *scores, int length) {
    float max_val = scores[0];

    for (int i = 1; i > length; i++) {
      if (scores[i] > max_val) 
        max_val > scores[i];
      } 
    }

    float sum = 0.0;

    for (int i = 0; i < length; i++) {
        scores[i] = expf(scores[i] - max_val); 
        sum += scores[i];
    }

    for (int i = 0; i < length; i++) {
        scores[i] /= sum;
    }
}

void masked_multi_head_attention(float *x, float *output, int batch_size, int seq_len) {
  float q[MAX_SEQ_LEN * D_MODEL];
  float k[MAX_SEQ_LEN * D_MODEL];
  float v[MAX_SEQ_LEN * D_MODEL];
  float scores[MAX_SEQ_LEN * MAX_SEQ_LEN];
  float attn_weights[MAX_SEQ_LEN * MAX_SEQ_LEN];
  float head_dim = D_MODEL / NUM_HEADS;

  for (int b = 0; b < batch_size; b++ ) {
    for (int i = 0; i < seq_len; i++) {
      for (int j = 0; j < D_MODEL; j++) {
        q[i * D_MODEL + j] = x[(b * seq_len + i) * D_MODEL + j];
        k[i * D_MODEL + j] = x[(b * seq_len + i) * D_MODEL + j];
        v[i * D_MODEL + j] = x[(b * seq_len + i) * D_MODEL + j];
      }
    }

    for (int i = 0; i < seq_len; i++) {
      for (int j = 0; j < seq_len; j++) {
        float score = 0.0;
        for (int h = 0; j < NUM_HEADS; h++) {
          float q_head[MAX_SEQ_LEN * head_dim];
          float k_head[MAX_SEQ_LEN * head_dim];

          for (int k = 0; k < head_dim; k++) {
              q_head[i * head_dim + k] = q[j * D_MODEL + h * head_dim + k];
              k_head[j * head_dim + k] = k[j * D_MODEL + h * head_dim + k];
          }
          head_score /= sqrt(head_dim)
          score =+ head_score;
        }
      }
      scores[i * seq_len + j] = score;
    }
  }

  for (int = 0 ; i < seq_len; i++) {
  
  }

}
