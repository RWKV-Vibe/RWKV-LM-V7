#include <assert.h>
#ifdef _FP32_
    using bf = float;
    #define to_float(u) (u)
    #define to_bf(u) (u)
#else
    #include <cuda_bf16.h>
    using bf = __nv_bfloat16;
    #define to_float(u) (__bfloat162float(u))
    #define to_bf(u) (__float2bfloat16_rn(u))
#endif

using i64 = long long int;
typedef bf * __restrict__ F_;
constexpr float W_SCALE = -0.6065306597f; // -exp(-0.5)
static_assert(_N_ == 128, "rwkv7_clampw128_v2 requires _N_=128");

//######################################################################################################

template<int N> __launch_bounds__(N,2)
__global__ void forward_kernel_preload(int T,int H,F_ r_,F_ w_,F_ k_,F_ v_,F_ a_,F_ b_,bf* __restrict__ y_,float* s__,float* __restrict__ sa_)
{
    const int bb=blockIdx.y, hh=blockIdx.x, i=threadIdx.x;
    float* __restrict__ s_ = s__ + i64(bb*H+hh) * i64((T/_CHUNK_LEN_)*N*N);
    float state[N];
#pragma unroll
    for (int j=0; j<N; ++j) {
        state[j] = 0.0f;
    }
    __shared__ float r[_CHUNK_LEN_][N];
    __shared__ float w[_CHUNK_LEN_][N];
    __shared__ float k[_CHUNK_LEN_][N];
    __shared__ float v[_CHUNK_LEN_][N];
    __shared__ float a[_CHUNK_LEN_][N];
    __shared__ float b[_CHUNK_LEN_][N];

    for (int t0 = 0; t0 < T; t0 += _CHUNK_LEN_)
    {
        __syncthreads();
#pragma unroll
        for (int tt=0; tt<_CHUNK_LEN_; ++tt) {
            const int idx = ((bb*T+t0+tt)*H+hh)*N+i;
            r[tt][i] = to_float(r_[idx]);
            w[tt][i] = __expf(W_SCALE / (1.0f + __expf(-to_float(w_[idx]))));
            k[tt][i] = to_float(k_[idx]);
            v[tt][i] = to_float(v_[idx]);
            a[tt][i] = to_float(a_[idx]);
            b[tt][i] = to_float(b_[idx]);
        }
        __syncthreads();

        for (int tt=0; tt<_CHUNK_LEN_; ++tt) {
            const int idx = ((bb*T+t0+tt)*H+hh)*N+i;

            float sa = 0.0f;
#pragma unroll
            for (int j=0; j<N; ++j) {
                sa += state[j] * a[tt][j];
            }
            sa_[idx] = sa;

            float vi = v[tt][i];
            float y=0.0f;
#pragma unroll
            for (int j=0; j<N; ++j) {
                float s = state[j];
                s = s * w[tt][j] + (sa * b[tt][j] + k[tt][j] * vi);
                y += s * r[tt][j];
                state[j] = s;
            }

            y_[idx] = to_bf(y);
        }

        {
            int base = (t0/_CHUNK_LEN_)*N*N;
#pragma unroll
            for (int j=0; j<N; ++j) {
                s_[base+i+j*N] = state[j];
            }
        }
    }
}

void cuda_forward_128_v2(int B,int T,int H,bf*r,bf*w,bf*k,bf*v,bf*a,bf*b,bf*y,float*s,float*sa)
{
    forward_kernel_preload<_N_><<<dim3(H,B),dim3(_N_)>>>(T,H,r,w,k,v,a,b,y,s,sa);
}

//######################################################################################################

template<int N> __launch_bounds__(256,1)
__global__ void backward_kernel_preload_split64(int T, int H, F_ r_, F_ w_, F_ k_, F_ v_, F_ a_, F_ b_, F_ dy_, float * __restrict__ s__, float * __restrict__ sa_, bf* dr_, bf* dw_, bf* dk_, bf* dv_, bf* da_, bf* db_)
{
    static_assert(N == 128, "split64 path is only for N=128");
    constexpr int TILE = 4;
    constexpr int SEG = 64;
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x, seg = threadIdx.y;
    int j0 = seg * SEG;
    float* __restrict__ s_ = s__ + i64(bb*H+hh) * i64((T/_CHUNK_LEN_)*N*N);

    float stateT[SEG] = {0}, dstate[SEG] = {0}, dstateT[SEG] = {0};
    __shared__ float r[TILE][N];
    __shared__ float w[TILE][N];
    __shared__ float ws[TILE][N];
    __shared__ float k[TILE][N];
    __shared__ float v[TILE][N];
    __shared__ float a[TILE][N];
    __shared__ float b[TILE][N];
    __shared__ float dy[TILE][N];
    __shared__ float sa[TILE][N];
    __shared__ float dSb_shared[N];
    __shared__ float partial[6][2][N];

    for (int t0 = T-_CHUNK_LEN_; t0 >= 0; t0 -= _CHUNK_LEN_)
    {
        {
            int base = (t0/_CHUNK_LEN_)*N*N + i*N + j0;
            const float4* s4 = (const float4*)(s_ + base);
#pragma unroll
            for (int j4 = 0; j4 < SEG/4; j4++) {
                float4 q = s4[j4];
                const int j = j4<<2;
                stateT[j+0] = q.x;
                stateT[j+1] = q.y;
                stateT[j+2] = q.z;
                stateT[j+3] = q.w;
            }
        }

        for (int subt=_CHUNK_LEN_-TILE; subt>=0; subt-=TILE) {
            __syncthreads();
            if (seg == 0) {
#pragma unroll
                for (int tt=0; tt<TILE; ++tt) {
                    int idx = bb*T*H*N + (t0+subt+tt)*H*N + hh * N + i;
                    r[tt][i] = to_float(r_[idx]);
                    float w_sig = 1.0f / (1.0f + __expf(-to_float(w_[idx])));
                    float wi = __expf(W_SCALE * w_sig);
                    ws[tt][i] = W_SCALE * wi * w_sig * (1.0f - w_sig);
                    w[tt][i] = wi;
                    k[tt][i] = to_float(k_[idx]);
                    v[tt][i] = to_float(v_[idx]);
                    a[tt][i] = to_float(a_[idx]);
                    b[tt][i] = to_float(b_[idx]);
                    dy[tt][i] = to_float(dy_[idx]);
                    sa[tt][i] = sa_[idx];
                }
            }
            __syncthreads();

            for (int tt=TILE-1; tt>=0; --tt) {
                int idx = bb*T*H*N + (t0+subt+tt)*H*N + hh * N + i;
                float ri = r[tt][i];
                float wi = w[tt][i];
                float ki = k[tt][i];
                float ai = a[tt][i];
                float bi = b[tt][i];
                float dyi = dy[tt][i];

                float dr = 0;
#pragma unroll
                for (int j = 0; j < SEG; j++) {
                    dr += stateT[j] * dy[tt][j0+j];
                }
                partial[5][seg][i] = dr;
                __syncthreads();
                if (seg == 0) {
                    dr_[idx] = to_bf(partial[5][0][i] + partial[5][1][i]);
                }

                float iwi = 1.0f / wi;
#pragma unroll
                for (int j = 0; j < SEG; j++) {
                    int jj = j0 + j;
                    stateT[j] = (stateT[j] - ki * v[tt][jj] - bi * sa[tt][jj]) * iwi;
                    dstate[j] += dyi * r[tt][jj];
                    dstateT[j] += ri * dy[tt][jj];
                }

                float dw = 0, dk = 0, dv = 0, db = 0, dSb = 0;
#pragma unroll
                for (int j = 0; j < SEG; j++) {
                    int jj = j0 + j;
                    dw += dstateT[j] * stateT[j];
                    dk += dstateT[j] * v[tt][jj];
                    dv += dstate[j] * k[tt][jj];
                    dSb += dstate[j] * b[tt][jj];
                    db += dstateT[j] * sa[tt][jj];
                }
                partial[0][seg][i] = dw;
                partial[1][seg][i] = dk;
                partial[2][seg][i] = dv;
                partial[3][seg][i] = db;
                partial[4][seg][i] = dSb;
                __syncthreads();
                if (seg == 0) {
                    float dw_sum = partial[0][0][i] + partial[0][1][i];
                    float dk_sum = partial[1][0][i] + partial[1][1][i];
                    float dv_sum = partial[2][0][i] + partial[2][1][i];
                    float db_sum = partial[3][0][i] + partial[3][1][i];
                    float dSb_sum = partial[4][0][i] + partial[4][1][i];
                    dw_[idx] = to_bf(dw_sum * ws[tt][i]);
                    dk_[idx] = to_bf(dk_sum);
                    dv_[idx] = to_bf(dv_sum);
                    db_[idx] = to_bf(db_sum);
                    dSb_shared[i] = dSb_sum;
                }
                __syncthreads();

                float da = 0;
#pragma unroll
                for (int j = 0; j < SEG; j++) {
                    int jj = j0 + j;
                    da += stateT[j] * dSb_shared[jj];
                }
                partial[0][seg][i] = da;
                __syncthreads();
                if (seg == 0) {
                    da_[idx] = to_bf(partial[0][0][i] + partial[0][1][i]);
                }

                float dSb_i = dSb_shared[i];
#pragma unroll
                for (int j = 0; j < SEG; j++) {
                    int jj = j0 + j;
                    dstate[j] = dstate[j] * w[tt][jj] + dSb_i * a[tt][jj];
                    dstateT[j] = dstateT[j] * wi + ai * dSb_shared[jj];
                }
            }
        }
    }
}

void cuda_backward_128_v2(int B, int T, int H, bf*r, bf*w, bf*k, bf*v, bf*a, bf*b, bf*dy, float*s, float*sa, bf*dr, bf*dw, bf*dk, bf*dv, bf*da, bf*db)
{
    assert(T%_CHUNK_LEN_ == 0);
    backward_kernel_preload_split64<_N_><<<dim3(H,B), dim3(_N_,2)>>>(T,H,r,w,k,v,a,b,dy,s,sa,dr,dw,dk,dv,da,db);
}
