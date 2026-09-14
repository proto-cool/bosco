#include "lif.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

struct lif_net {
    int32_t n;
    int64_t nnz;
    int64_t *indptr;
    int32_t *indices;
    double *w;

    lif_params p;
    double e_m, e_s, c_g;   /* exact-integration coefficients */
    int32_t rfc_steps;      /* default refractory length in steps */
    int32_t dly_steps;

    double *v, *g, *x, *u;  /* u: per-presynaptic-neuron STD utilisation */
    double *theta;          /* adaptive threshold offset */
    double sfa_b, e_sfa;
    double std_u, e_rec;    /* STD utilisation and per-step recovery factor */
    int32_t *rfc_len;       /* per-neuron refractory length (0 for driven inputs) */
    int32_t *rfc_left;      /* steps of refractoriness remaining */
    uint8_t *spiked;        /* spiked this step */

    /* delay ring: dly_steps slots, each holds up to n ids */
    int32_t *ring;
    int32_t *ring_cnt;
    int32_t ring_pos;

    int32_t n_in;
    int32_t *in_idx;
    double *in_p, *in_w;

    uint64_t rng;
    int64_t step;
    int64_t *counts;
};

/* splitmix64 */
static inline uint64_t next_u64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline double next_unit(uint64_t *s) {
    return (double)(next_u64(s) >> 11) * (1.0 / 9007199254740992.0);
}

lif_net *lif_create(int32_t n, const int64_t *indptr, const int32_t *indices,
                    const double *w, const lif_params *p) {
    lif_net *net = calloc(1, sizeof *net);
    if (!net) return NULL;
    net->n = n;
    net->nnz = indptr[n];
    net->indptr = malloc((size_t)(n + 1) * sizeof(int64_t));
    net->indices = malloc((size_t)net->nnz * sizeof(int32_t));
    net->w = malloc((size_t)net->nnz * sizeof(double));
    memcpy(net->indptr, indptr, (size_t)(n + 1) * sizeof(int64_t));
    memcpy(net->indices, indices, (size_t)net->nnz * sizeof(int32_t));
    memcpy(net->w, w, (size_t)net->nnz * sizeof(double));
    net->p = *p;
    net->e_m = exp(-p->dt_ms / p->tau_m);
    net->e_s = exp(-p->dt_ms / p->tau_s);
    net->c_g = (p->tau_s / (p->tau_m - p->tau_s)) * (net->e_m - net->e_s);
    net->rfc_steps = (int32_t)llround(p->t_rfc / p->dt_ms);
    net->dly_steps = (int32_t)llround(p->t_dly / p->dt_ms);
    if (net->dly_steps < 1) net->dly_steps = 1;
    net->std_u = p->std_u;
    net->e_rec = (p->std_u > 0.0 && p->std_tau_rec > 0.0) ? exp(-p->dt_ms / p->std_tau_rec) : 1.0;
    net->v = malloc((size_t)n * sizeof(double));
    net->g = malloc((size_t)n * sizeof(double));
    net->x = malloc((size_t)n * sizeof(double));
    net->theta = malloc((size_t)n * sizeof(double));
    net->sfa_b = p->sfa_b;
    net->e_sfa = (p->sfa_b > 0.0 && p->sfa_tau > 0.0) ? exp(-p->dt_ms / p->sfa_tau) : 1.0;
    net->u = malloc((size_t)n * sizeof(double));
    for (int32_t i = 0; i < n; i++) net->u[i] = p->std_u;
    net->rfc_len = malloc((size_t)n * sizeof(int32_t));
    net->rfc_left = malloc((size_t)n * sizeof(int32_t));
    net->spiked = malloc((size_t)n);
    net->ring = malloc((size_t)net->dly_steps * (size_t)n * sizeof(int32_t));
    net->ring_cnt = malloc((size_t)net->dly_steps * sizeof(int32_t));
    net->counts = malloc((size_t)n * sizeof(int64_t));
    net->n_in = 0;
    net->in_idx = NULL; net->in_p = NULL; net->in_w = NULL;
    lif_reset(net, 0);
    return net;
}

void lif_free(lif_net *net) {
    if (!net) return;
    free(net->indptr); free(net->indices); free(net->w);
    free(net->v); free(net->g); free(net->x); free(net->theta); free(net->u); free(net->rfc_len); free(net->rfc_left);
    free(net->spiked); free(net->ring); free(net->ring_cnt); free(net->counts);
    free(net->in_idx); free(net->in_p); free(net->in_w);
    free(net);
}

void lif_reset(lif_net *net, uint64_t seed) {
    int32_t n = net->n;
    for (int32_t i = 0; i < n; i++) {
        net->v[i] = net->p.v0;
        net->g[i] = 0.0;
        net->x[i] = 1.0;
        net->theta[i] = 0.0;
        net->rfc_len[i] = net->rfc_steps;
        net->rfc_left[i] = 0;
        net->spiked[i] = 0;
        net->counts[i] = 0;
    }
    for (int32_t k = 0; k < net->n_in; k++) net->rfc_len[net->in_idx[k]] = 0;
    memset(net->ring_cnt, 0, (size_t)net->dly_steps * sizeof(int32_t));
    net->ring_pos = 0;
    net->rng = seed ^ 0xD1B54A32D192ED03ULL;
    /* warm the generator so seed 0 is not the raw state */
    for (int i = 0; i < 4; i++) (void)next_u64(&net->rng);
    net->step = 0;
}

void lif_set_weights(lif_net *net, const int64_t *edge_idx, const double *w, int64_t m) {
    for (int64_t k = 0; k < m; k++) net->w[edge_idx[k]] = w[k];
}
void lif_get_weights(const lif_net *net, double *w_out) {
    memcpy(w_out, net->w, (size_t)net->nnz * sizeof(double));
}
int64_t lif_nnz(const lif_net *net) { return net->nnz; }

void lif_set_inputs(lif_net *net, int32_t m, const int32_t *idx,
                    const double *rate_hz, const double *weight_mv) {
    /* restore refractory period of previously driven neurons */
    for (int32_t k = 0; k < net->n_in; k++) net->rfc_len[net->in_idx[k]] = net->rfc_steps;
    free(net->in_idx); free(net->in_p); free(net->in_w);
    net->n_in = m;
    net->in_idx = malloc((size_t)m * sizeof(int32_t));
    net->in_p = malloc((size_t)m * sizeof(double));
    net->in_w = malloc((size_t)m * sizeof(double));
    for (int32_t k = 0; k < m; k++) {
        net->in_idx[k] = idx[k];
        net->in_p[k] = rate_hz[k] * net->p.dt_ms * 1e-3;
        net->in_w[k] = weight_mv[k];
        net->rfc_len[idx[k]] = 0;
    }
}

int64_t lif_run(lif_net *net, int64_t n_steps, int32_t *t_out, int32_t *id_out,
                int64_t max_spikes) {
    const int32_t n = net->n;
    const double e_m = net->e_m, e_s = net->e_s, c_g = net->c_g;
    const double v0 = net->p.v0, v_th = net->p.v_th, v_rst = net->p.v_rst;
    double *v = net->v, *g = net->g, *x = net->x, *theta = net->theta;
    const double e_rec = net->e_rec;
    const double sfa_b = net->sfa_b, e_sfa = net->e_sfa;
    const int sfa_on = (sfa_b > 0.0);
    const double *u = net->u;
    const int std_on = (net->std_u > 0.0);
    int32_t *rfc_left = net->rfc_left;
    const int32_t *rfc_len = net->rfc_len;
    uint8_t *spiked = net->spiked;
    const int32_t D = net->dly_steps;
    int64_t total = 0;

    for (int64_t s = 0; s < n_steps; s++) {
        const int32_t t = (int32_t)net->step;
        /* 0. synaptic resource recovery (all neurons, including refractory) */
        if (std_on) {
            for (int32_t i = 0; i < n; i++) x[i] = 1.0 - (1.0 - x[i]) * e_rec;
        }
        /* 1. state update (exact for the linear system) unless refractory; threshold
              relaxation for all.  Branch-free flushes so the loop vectorises; tiny values
              are flushed to zero because exponential decay reaches the denormal range
              after ~40 k steps and denormal arithmetic is ~100x slower. */
        for (int32_t i = 0; i < n; i++) {
            const int free = (rfc_left[i] == 0);
            const double gi = g[i];
            double u = (v[i] - v0) * e_m + gi * c_g;
            double gn = gi * e_s;
            u = (fabs(u) < 1e-30) ? 0.0 : u;
            gn = (fabs(gn) < 1e-30) ? 0.0 : gn;
            v[i] = free ? v0 + u : v[i];
            g[i] = free ? gn : gi;
            rfc_left[i] = free ? 0 : rfc_left[i] - 1;
            if (sfa_on) {
                double th = theta[i] * e_sfa;
                theta[i] = (th < 1e-30) ? 0.0 : th;
            }
        }
        /* 2. threshold (refractory neurons were skipped above and cannot spike:
              their v is frozen at v_rst) */
        int32_t *slot = net->ring + (size_t)net->ring_pos * (size_t)n;
        int32_t cnt = 0;
        for (int32_t i = 0; i < n; i++) {
            if (rfc_left[i] == 0 && v[i] > v_th + theta[i]) {
                spiked[i] = 1;
                slot[cnt++] = i;
                net->counts[i]++;
                if (total < max_spikes) { t_out[total] = t; id_out[total] = i; }
                total++;
            } else {
                spiked[i] = 0;
            }
        }
        net->ring_cnt[net->ring_pos] = cnt;
        /* 3a. deliver spikes emitted D steps ago */
        {
            int32_t dpos = (net->ring_pos + 1) % D;   /* the oldest slot */
            int32_t dcnt = net->ring_cnt[dpos];
            const int32_t *dslot = net->ring + (size_t)dpos * (size_t)n;
            for (int32_t k = 0; k < dcnt; k++) {
                int32_t pre = dslot[k];
                int64_t a = net->indptr[pre], b = net->indptr[pre + 1];
                if (std_on) {
                    const double xp = x[pre];
                    for (int64_t e = a; e < b; e++) g[net->indices[e]] += net->w[e] * xp;
                    x[pre] = xp - u[pre] * xp;
                } else {
                    for (int64_t e = a; e < b; e++) g[net->indices[e]] += net->w[e];
                }
            }
            net->ring_cnt[dpos] = 0;
        }
        /* 3b. Poisson inputs (Bernoulli per step), fixed order */
        for (int32_t k = 0; k < net->n_in; k++) {
            if (next_unit(&net->rng) < net->in_p[k]) v[net->in_idx[k]] += net->in_w[k];
        }
        /* 4. reset (+ adaptation; driven inputs, which have rfc_len 0, are exempt) */
        for (int32_t k = 0; k < cnt; k++) {
            int32_t i = slot[k];
            v[i] = v_rst;
            g[i] = 0.0;
            rfc_left[i] = rfc_len[i];
            if (sfa_on && rfc_len[i] > 0) theta[i] += sfa_b;
        }
        net->ring_pos = (net->ring_pos + 1) % D;
        net->step++;
    }
    return total;
}

void lif_spike_counts(const lif_net *net, int64_t *counts) {
    memcpy(counts, net->counts, (size_t)net->n * sizeof(int64_t));
}
int64_t lif_state_size(const lif_net *net) {
    int64_t n = net->n, D = net->dly_steps;
    return 4 * n * (int64_t)sizeof(double) + (n + D * n + D + 1) * (int64_t)sizeof(int32_t)
         + (int64_t)sizeof(uint64_t) + (int64_t)sizeof(int64_t);
}
void lif_get_state(const lif_net *net, void *buf) {
    int64_t n = net->n, D = net->dly_steps;
    char *p = buf;
    memcpy(p, net->v, n * sizeof(double)); p += n * sizeof(double);
    memcpy(p, net->g, n * sizeof(double)); p += n * sizeof(double);
    memcpy(p, net->x, n * sizeof(double)); p += n * sizeof(double);
    memcpy(p, net->theta, n * sizeof(double)); p += n * sizeof(double);
    memcpy(p, net->rfc_left, n * sizeof(int32_t)); p += n * sizeof(int32_t);
    memcpy(p, net->ring, D * n * sizeof(int32_t)); p += D * n * sizeof(int32_t);
    memcpy(p, net->ring_cnt, D * sizeof(int32_t)); p += D * sizeof(int32_t);
    memcpy(p, &net->ring_pos, sizeof(int32_t)); p += sizeof(int32_t);
    memcpy(p, &net->rng, sizeof(uint64_t)); p += sizeof(uint64_t);
    memcpy(p, &net->step, sizeof(int64_t));
}
void lif_set_state(lif_net *net, const void *buf) {
    int64_t n = net->n, D = net->dly_steps;
    const char *p = buf;
    memcpy(net->v, p, n * sizeof(double)); p += n * sizeof(double);
    memcpy(net->g, p, n * sizeof(double)); p += n * sizeof(double);
    memcpy(net->x, p, n * sizeof(double)); p += n * sizeof(double);
    memcpy(net->theta, p, n * sizeof(double)); p += n * sizeof(double);
    memcpy(net->rfc_left, p, n * sizeof(int32_t)); p += n * sizeof(int32_t);
    memcpy(net->ring, p, D * n * sizeof(int32_t)); p += D * n * sizeof(int32_t);
    memcpy(net->ring_cnt, p, D * sizeof(int32_t)); p += D * sizeof(int32_t);
    memcpy(&net->ring_pos, p, sizeof(int32_t)); p += sizeof(int32_t);
    memcpy(&net->rng, p, sizeof(uint64_t)); p += sizeof(uint64_t);
    memcpy(&net->step, p, sizeof(int64_t));
}
void lif_recover(lif_net *net, double ms) {
    const double fr = (net->std_u > 0.0 && net->p.std_tau_rec > 0.0) ? exp(-ms / net->p.std_tau_rec) : 0.0;
    const double fs = (net->sfa_b > 0.0 && net->p.sfa_tau > 0.0) ? exp(-ms / net->p.sfa_tau) : 0.0;
    for (int32_t i = 0; i < net->n; i++) {
        net->x[i] = 1.0 - (1.0 - net->x[i]) * fr;
        net->theta[i] *= fs;
        if (net->theta[i] < 1e-30) net->theta[i] = 0.0;
    }
}
void lif_set_std_u(lif_net *net, const double *u) {
    memcpy(net->u, u, (size_t)net->n * sizeof(double));
}
void lif_get_x(const lif_net *net, double *x_out) {
    memcpy(x_out, net->x, (size_t)net->n * sizeof(double));
}
void lif_get_v(const lif_net *net, double *v_out) {
    memcpy(v_out, net->v, (size_t)net->n * sizeof(double));
}
int64_t lif_step(const lif_net *net) { return net->step; }
